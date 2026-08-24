import os
import time
import math
import gc
import json
import shutil
import subprocess
import argparse
import numpy as np
import torch
import torch.nn as nn
from model import nanoSubQ, nanoSubQConfig

data_dir = 'data'
out_dir = 'out'
kaggle_sync_dir = 'kaggle_sync'   # staging dir for the Kaggle Dataset backup

ap = argparse.ArgumentParser(description="Train nanoSubQ (near-linear content-routed attention).")
ap.add_argument('--resume', type=str, default=None,
                help="checkpoint .pt, or a dir containing ckpt_step_*.pt, to resume from (continue training)")
ap.add_argument('--kaggle-dataset', type=str, default=None,
                help="owner/dataset to durably back up the latest checkpoint to after each eval "
                     "(auto-creates the dataset on first push; needs KAGGLE_USERNAME/KAGGLE_KEY env "
                     "from notebook Secrets and the `kaggle` CLI). Resume later with: "
                     "kaggle datasets download <owner>/<dataset> -p restore && unzip -o restore/*.zip -d restore "
                     "then: python train.py --resume restore/latest.pt")
args = ap.parse_args()


def push_to_kaggle(ckpt_path, iter_num):
    """Durably back up the newest checkpoint to a Kaggle Dataset (survives session reaping).

    Only 'latest.pt' is synced so each upload stays a single ~578MB file. Non-fatal: a failed
    push is logged and training keeps going (the local out/ checkpoint is still the source of
    truth until you download this one)."""
    if shutil.which('kaggle') is None:
        print("  !! kaggle CLI not found -- install it with `pip install kaggle` to enable dataset backup")
        return
    os.makedirs(kaggle_sync_dir, exist_ok=True)
    latest = os.path.join(kaggle_sync_dir, 'latest.pt')
    if os.path.exists(latest):
        os.remove(latest)
    try:
        os.link(ckpt_path, latest)          # hardlink: instant, no extra disk copy
    except OSError:
        shutil.copyfile(ckpt_path, latest)
    meta = os.path.join(kaggle_sync_dir, 'dataset-metadata.json')
    if not os.path.exists(meta):
        with open(meta, 'w') as f:
            json.dump({'id': None, 'title': args.kaggle_dataset.split('/')[-1], 'isPrivate': True}, f)
    with open(meta) as f:
        is_new = json.load(f).get('id') is None
    cmd = ['kaggle', 'datasets', 'create' if is_new else 'version',
           '-p', kaggle_sync_dir, '-m', f'step {iter_num}']
    print(f"  backing up latest checkpoint to Kaggle dataset {args.kaggle_dataset} ...", flush=True)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        tail = (r.stdout or r.stderr or '').strip()[-400:]
        if r.returncode == 0:
            print(f"  kaggle backup OK: {tail}", flush=True)
        else:
            print(f"  !! kaggle push reported failure (non-fatal, continuing): {tail}", flush=True)
    except Exception as e:
        print(f"  !! kaggle push failed (non-fatal, continuing): {e}", flush=True)

micro_batch_size = 16
gradient_accumulation_steps = 2
block_size = 1024

# data/prepare.py writes uint32; read as uint32 (reading as uint16 corrupts every other token to 0)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint32, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint32, mode='r')

# ~5h on 2x T4 (fp16): 16000 iters ≈ 524M tokens (~57% of the 922M-token train.bin) at ~1.1 s/step.
# Experiment budget. Tune with: iters ≈ hours*3600/(s per step).
max_iters = 16000
eval_interval = 2500
log_interval = 20
eval_iters = 50
warmup_iters = 1000

learning_rate = 5.0e-4
min_lr = 5.0e-5
weight_decay = 0.1
max_grad_norm = 1.0      # the recipe the ssa repo trains with (was 0.5)

# Research knob: anneal the routing temperature beta over training (4.0 -> 2.0).
# None = keep config.beta fixed (the repo's measured optimum ~2).
# Enabled by default: start more variance-seeking, settle at the measured optimum ~2.
beta_anneal = (4.0, 2.0)

config = nanoSubQConfig(
    vocab_size=50304,
    max_seq_len=1024,
    d_model=384,
    num_layers=6,
    num_q_heads=6,
    num_kv_heads=2,
    block=128,         # routing block size; must be a multiple of 128 for the CUDA flex kernel (1024 // 128 = 8 blocks)
    top_c=4,           # ~50% of causally-visible blocks selected + local window
    local=1,
    beta=2.0,          # cumulant routing temperature
    attn_impl='flex',  # O(n*kappa) fused kernel; 'masked' = O(n^2) exact reference
    label_smoothing=0.1,  # soften CE target -> every vocab token gets a gradient
    dropout=0.0,
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# bf16 needs sm_80+ (A100/L4/...). T4 is sm_75 -> no bf16 hardware; must use fp16 + GradScaler.
# Check the capability directly: torch.cuda.is_bf16_supported() can wrongly report True on T4.
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    ptdtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
else:
    ptdtype = torch.float16
scaler = torch.amp.GradScaler('cuda', enabled=(ptdtype == torch.float16))

print(f"Using device: {device} | Precision: {ptdtype}")
os.makedirs(out_dir, exist_ok=True)

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (micro_batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

raw_model = nanoSubQ(config)

def configure_optimizer(model, weight_decay, learning_rate):
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "ln" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay, "lr": learning_rate},
        {"params": no_decay_params, "weight_decay": 0.0, "lr": learning_rate},
    ]

    return torch.optim.AdamW(optim_groups, betas=(0.9, 0.95))

optimizer = configure_optimizer(raw_model, weight_decay, learning_rate)

if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    print(f"Enabling DataParallel across {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(raw_model).to(device)
else:
    model = raw_model.to(device)

# Pre-compile the flex kernel (forward AND backward) on every GPU, single-threaded.
# torch.compile + nn.DataParallel is not thread-safe during the FIRST compile: DP runs the
# replicas in worker threads while the main thread compiles the backward, and torch's
# compilation-metrics context is one GLOBAL (not thread-local), so the worker-thread forward
# compile leaves 'is_forward' set and the first backward compile raises
# "Metric(s) {'is_forward'} have already been set in the current context".
# Warming up fwd+bwd per device with the exact per-replica batch makes every training-loop call
# a compile-cache hit, so no compilation ever happens under DataParallel. (~1-2 min one-time.)
if torch.cuda.is_available() and torch.cuda.device_count() > 1 and config.attn_impl == 'flex':
    warm_batch = max(1, micro_batch_size // torch.cuda.device_count())   # what DP gives each replica
    for gi in range(torch.cuda.device_count()):
        with torch.cuda.device(gi):
            w = nanoSubQ(config).to(f'cuda:{gi}')
            wx = torch.randint(0, config.vocab_size, (warm_batch, block_size), device=f'cuda:{gi}')
            wy = torch.randint(0, config.vocab_size, (warm_batch, block_size), device=f'cuda:{gi}')
            with torch.amp.autocast(device_type='cuda', dtype=ptdtype):
                _, wl, *_ = w(wx, targets=wy)
            wl.sum().backward()       # also compiles the flex backward graph
            del w
        torch.cuda.empty_cache()
    print(f"pre-compiled flex kernel (fwd+bwd) on {torch.cuda.device_count()} GPUs")

# Resume: restore model weights, optimizer (AdamW momentum), scaler and step counter from a
# previous run. If the session dies (e.g. Kaggle's 12h cap), rerun with `--resume out` to continue
# exactly where it stopped — LR schedule position and optimizer state intact. max_iters can be
# raised to train further. NOTE: checkpoint must survive the session end (see 'out/' caveat).
start_iter = 0
if args.resume is not None:
    ckpt_path = args.resume
    if os.path.isdir(ckpt_path):
        files = sorted(
            (f for f in os.listdir(ckpt_path) if f.startswith('ckpt_step_') and f.endswith('.pt')),
            key=lambda f: int(f[len('ckpt_step_'):-len('.pt')]),
        )
        assert files, f"no ckpt_step_*.pt found in {ckpt_path}"
        ckpt_path = os.path.join(ckpt_path, files[-1])
    print(f"resuming from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    raw_model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scaler.load_state_dict(ckpt['scaler'])
    start_iter = int(ckpt['iter'])
    print(f"resumed at step {start_iter}/{max_iters}")

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

model.train()
t0 = time.time()

for iter_num in range(start_iter + 1, max_iters + 1):
    lr = get_lr(iter_num)

    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = lr

    if beta_anneal is not None:
        prog = max(0.0, min(1.0, (iter_num - warmup_iters) / (max_iters - warmup_iters)))
        beta = beta_anneal[0] - prog * (beta_anneal[0] - beta_anneal[1])
        for layer in raw_model.layers:
            layer.attn.beta = beta
    else:
        beta = config.beta

    optimizer.zero_grad(set_to_none=True)
    accum_loss = 0.0
    accum_sparsity = 0.0
    accum_entropy = 0.0
    accum_load = 0.0

    for micro_step in range(gradient_accumulation_steps):
        x, y = get_batch('train')

        with torch.amp.autocast(device_type='cuda', dtype=ptdtype):
            logits, ce_loss, sparsity, entropy, load = model(x, targets=y)
            loss_scaled = ce_loss.mean() / gradient_accumulation_steps

        accum_loss += ce_loss.mean().item()
        accum_sparsity += sparsity.mean().item()
        accum_entropy += entropy.mean().item()
        accum_load += load.mean().item()

        if ptdtype == torch.float16:
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

    if ptdtype == torch.float16:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

    if iter_num % log_interval == 0:
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        avg_loss = accum_loss / gradient_accumulation_steps
        avg_sparsity = accum_sparsity / gradient_accumulation_steps
        avg_entropy = accum_entropy / gradient_accumulation_steps
        avg_load = accum_load / gradient_accumulation_steps
        print(f"step {iter_num:5d}/{max_iters} | loss {avg_loss:.4f} | sparsity {avg_sparsity:.3f} | "
              f"entropy {avg_entropy:.3f} | load {avg_load:.2f} | beta {beta:.2f} | lr {lr:.6f} | "
              f"time {dt*1000/log_interval:.2f}ms/step")

    if iter_num % eval_interval == 0 or iter_num == max_iters:
        model.eval()
        with torch.no_grad():
            val_losses = []
            for _ in range(eval_iters):
                x_v, y_v = get_batch('val')
                with torch.amp.autocast(device_type='cuda', dtype=ptdtype):
                    _, v_loss, *_ = model(x_v, targets=y_v)
                val_losses.append(v_loss.mean().item())
            mean_val_loss = sum(val_losses) / len(val_losses)
            print(f"\n--- EVAL @ Step {iter_num}/{max_iters} | Validation Loss: {mean_val_loss:.4f} ---\n")

            ckpt_path = os.path.join(out_dir, f'ckpt_step_{iter_num}.pt')
            torch.save({
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'iter': iter_num,
                'config': config,
            }, ckpt_path)
            print(f"  saved {ckpt_path}")
            if args.kaggle_dataset is not None:
                push_to_kaggle(ckpt_path, iter_num)

        torch.cuda.empty_cache()
        gc.collect()
        model.train()

print("Training finished successfully!")
