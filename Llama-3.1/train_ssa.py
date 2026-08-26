"""
Train SSA-LLaMA 3.1 8B — multi-node DDP with content-routed sparse attention.

Launch with torchrun:
    # Single node, 8 GPUs
    torchrun --nproc_per_node=8 train_ssa.py

    # 8 nodes × 8 GPUs = 64 GPUs
    torchrun --nproc_per_node=8 --nnodes=8 --node_rank=$NODE_RANK \
             --master_addr=$MASTER_ADDR --master_port=29500 train_ssa.py
"""
from __future__ import annotations
import os, sys, time, math, gc, argparse
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.dirname(__file__))
from ssa_llama import SSA_LlamaAttention


# ── Distributed helpers ────────────────────────────────────────────────────────
def setup_distributed():
    """Initialize DDP process group from torchrun env vars."""
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def is_main(rank):
    return rank == 0


def log(rank, *a, **kw):
    if is_main(rank):
        print(*a, **kw)


# ── Config ─────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser(description="Train SSA-LLaMA 3.1 8B (multi-node DDP)")
ap.add_argument("--model-id", type=str, default="meta-llama/Llama-3.1-8B")
ap.add_argument("--data-dir", type=str, default="data")
ap.add_argument("--out-dir", type=str, default="out_ssa")
ap.add_argument("--resume", type=str, default=None)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--min-lr", type=float, default=2e-6)
ap.add_argument("--warmup-iters", type=int, default=200)
ap.add_argument("--max-iters", type=int, default=10000)
ap.add_argument("--eval-interval", type=int, default=500)
ap.add_argument("--log-interval", type=int, default=10)
ap.add_argument("--eval-iters", type=int, default=20)
ap.add_argument("--batch-size", type=int, default=2,
                help="Micro batch size per GPU (per forward pass)")
ap.add_argument("--grad-accum", type=int, default=8,
                help="Gradient accumulation steps per GPU")
ap.add_argument("--block-size", type=int, default=2048)
ap.add_argument("--weight-decay", type=float, default=0.1)
ap.add_argument("--max-grad-norm", type=float, default=1.0)
ap.add_argument("--beta-anneal", type=str, default="4.0,2.0")
ap.add_argument("--top-c", type=int, default=4)
ap.add_argument("--local-w", type=int, default=1)
ap.add_argument("--grad-checkpoint", action="store_true",
                help="Enable gradient checkpointing (saves ~40% memory)")
args = ap.parse_args()


# ── Data ───────────────────────────────────────────────────────────────────────
def load_data(data_dir, split):
    path = os.path.join(data_dir, f"{split}.bin")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}.")
    return np.memmap(path, dtype=np.uint32, mode='r')


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


# ── Model ──────────────────────────────────────────────────────────────────────
def build_model(model_id, rank):
    from transformers import LlamaForCausalLM

    log(rank, f"Loading {model_id} ...")
    model = LlamaForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map={"": rank},  # load directly to this GPU
        attn_implementation="eager",
    )

    # Gradient checkpointing (before freezing, so it applies to all layers)
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
        log(rank, "Gradient checkpointing enabled")

    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # Swap attention layers with SSA
    for i, layer in enumerate(model.model.layers):
        ssa_attn = SSA_LlamaAttention(model.config, layer_idx=i)

        # Copy pre-trained weights
        ssa_attn.q_proj.weight.data.copy_(layer.self_attn.q_proj.weight.data)
        ssa_attn.k_proj.weight.data.copy_(layer.self_attn.k_proj.weight.data)
        ssa_attn.v_proj.weight.data.copy_(layer.self_attn.v_proj.weight.data)
        ssa_attn.o_proj.weight.data.copy_(layer.self_attn.o_proj.weight.data)

        # Unfreeze attention projections + routing (routing is randomly initialized)
        ssa_attn.q_proj.weight.requires_grad = True
        ssa_attn.k_proj.weight.requires_grad = True
        ssa_attn.v_proj.weight.requires_grad = True
        ssa_attn.o_proj.weight.requires_grad = True

        ssa_attn.attn_impl = 'masked'
        ssa_attn.top_c = args.top_c
        ssa_attn.local = args.local_w
        layer.self_attn = ssa_attn

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(rank, f"Parameters: {n_total/1e9:.2f}B total, {n_train/1e6:.1f}M trainable ({n_train/n_total*100:.1f}%)")

    return model


# ── Training ───────────────────────────────────────────────────────────────────
def get_lr(step, warmup, max_iters, lr, min_lr):
    if step < warmup:
        return lr * (step + 1) / warmup
    if step > max_iters:
        return min_lr
    decay = (step - warmup) / (max_iters - warmup)
    return min_lr + 0.5 * (1 + math.cos(math.pi * decay)) * (lr - min_lr)


def evaluate(model, data, block_size, eval_iters, batch_size, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            out = model(x, labels=y)
            losses.append(out.loss.item())
    model.train()
    return np.mean(losses)


def train(model, train_data, val_data, rank, world_size):
    device = torch.device(f"cuda:{rank}")
    raw_model = model.module  # unwrap DDP

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95),
    )

    start_iter = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        raw_model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_iter = ckpt['iter']
        log(rank, f"Resumed from {args.resume} at step {start_iter}")

    if is_main(rank):
        os.makedirs(args.out_dir, exist_ok=True)

    # Beta annealing
    beta_parts = [float(x) for x in args.beta_anneal.split(",")]
    beta_schedule = tuple(beta_parts) if len(beta_parts) == 2 and args.beta_anneal != "none" else None

    scaler = torch.amp.GradScaler('cuda')

    model.train()
    t0 = time.time()

    for step in range(start_iter + 1, args.max_iters + 1):
        lr = get_lr(step, args.warmup_iters, args.max_iters, args.lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Anneal routing beta
        if beta_schedule:
            prog = max(0.0, min(1.0, (step - args.warmup_iters) / (args.max_iters - args.warmup_iters)))
            beta = beta_schedule[0] - prog * (beta_schedule[0] - beta_schedule[1])
            for layer in raw_model.model.layers:
                if hasattr(layer.self_attn, 'set_beta'):
                    layer.self_attn.set_beta(beta)

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro in range(args.grad_accum):
            x, y = get_batch(train_data, args.block_size, args.batch_size, device)
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = model(x, labels=y)
                loss = out.loss / args.grad_accum

            scaler.scale(loss).backward()
            accum_loss += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        # Logging (rank 0 only)
        if step % args.log_interval == 0 and is_main(rank):
            dt = time.time() - t0
            t0 = time.time()
            beta_val = beta_schedule[0] if not beta_schedule else beta
            effective_batch = args.batch_size * args.grad_accum * world_size
            print(f"step {step:5d}/{args.max_iters} | loss {accum_loss:.4f} | "
                  f"beta {beta_val:.2f} | lr {lr:.2e} | eff_batch {effective_batch} | "
                  f"{dt*1000/args.log_interval:.0f}ms/step")

        # Eval + checkpoint (rank 0 only)
        if (step % args.eval_interval == 0 or step == args.max_iters) and is_main(rank):
            val_loss = evaluate(model, val_data, args.block_size, args.eval_iters, args.batch_size, device)
            print(f"\n--- EVAL @ Step {step} | val loss: {val_loss:.4f} ---\n")

            ckpt_path = os.path.join(args.out_dir, f'ckpt_step_{step}.pt')
            torch.save({
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iter': step,
                'config': vars(args),
            }, ckpt_path)
            print(f"  saved {ckpt_path}")
            torch.cuda.empty_cache()
            gc.collect()

        # Sync ranks at eval boundaries
        if step % args.eval_interval == 0:
            dist.barrier()


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{rank}")

    log(rank, f"Training on {world_size} GPUs "
        f"({torch.cuda.get_device_name(rank)} on rank {rank})")

    train_data = load_data(args.data_dir, 'train')
    val_data = load_data(args.data_dir, 'val')
    log(rank, f"Data: train {len(train_data):,} tokens, val {len(val_data):,} tokens")

    model = build_model(args.model_id, rank)

    # Wrap with DDP
    model = DDP(model, device_ids=[rank], output_device=rank)

    train(model, train_data, val_data, rank, world_size)

    dist.destroy_process_group()
