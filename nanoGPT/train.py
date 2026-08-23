"""
This training script supports single GPU and DDP training for nanoSubQ using
the SmolTalk ChatML dataset with Supervised Loss Masking (ignore_index = -100).

To run on a single GPU:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 GPUs:
$ torchrun --standalone --nproc_per_node=4 train.py
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# I/O
out_dir = 'out'
eval_interval = 500
log_interval = 10
eval_iters = 100
eval_only = False 
always_save_checkpoint = True 
init_from = 'scratch' # 'scratch' or 'resume'

# wandb logging
wandb_log = False 
wandb_project = 'smoltalk-nanosubq'
wandb_run_name = 'sparse-gpt'

# Data config
dataset = 'smoltalk'
gradient_accumulation_steps = 4 
batch_size = 8 
block_size = 1024

# Model config
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 
bias = False 

# Optimizer settings
learning_rate = 6e-4 
max_iters = 50000 
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 

# Learning rate decay settings
decay_lr = True 
warmup_iters = 1000 
lr_decay_iters = 50000 
min_lr = 6e-5 

# ChatML Tokens & Masking Settings
IM_START = 50257
IM_END = 50258

# DDP settings
backend = 'nccl' 
device = 'cuda' 
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' 
compile = True 
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) if os.path.exists('configurator.py') else None
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# DDP setup
ddp = int(os.environ.get('RANK', -1)) != -1 
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 
    seed_offset = ddp_rank 
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
if master_process:
    print(f"Tokens per iteration will be: {tokens_per_iter:,}")
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cudnn.allow_tf32 = True 
device_type = 'cuda' if 'cuda' in device else 'cpu' 
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Data loader with Supervised Loss Masking (-100)
data_dir = os.path.join('data', dataset) if os.path.exists(os.path.join('data', dataset)) else 'data'

def get_batch(split):
    data_file = 'train.bin' if split == 'train' else 'val.bin'
    data_path = os.path.join(data_dir, data_file)
    
    # uint32 array support for high token IDs (ChatML: 50257, 50258)
    data = np.memmap(data_path, dtype=np.uint32, mode='r')
    
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    # Construct Supervised Mask Targets (ignore_index = -100 for user prompts)
    targets = torch.full_like(y, fill_value=-100)
    
    for b in range(batch_size):
        tokens = x[b].tolist()
        in_assistant_response = False
        
        for t in range(len(tokens)):
            # Turn on loss computation after encountering <|im_start|>assistant
            if tokens[t] == IM_START:
                in_assistant_response = True
            elif tokens[t] == IM_END:
                if in_assistant_response:
                    targets[b, t] = y[b, t] # Keep the <|im_end|> token target
                in_assistant_response = False
            
            if in_assistant_response:
                targets[b, t] = y[b, t]

    if device_type == 'cuda':
        x, targets = x.pin_memory().to(device, non_blocking=True), targets.pin_memory().to(device, non_blocking=True)
    else:
        x, targets = x.to(device), targets.to(device)
        
    return x, targets

iter_num = 0
best_val_loss = 1e9

# Model setup
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=50259, dropout=dropout) 

if init_from == 'scratch':
    print("Initializing a new nanoSubQ model from scratch...")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

model.to(device)

# GradScaler setup
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# Optimizer setup
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None 

if compile:
    print("Compiling the model...")
    model = torch.compile(model) 

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# Training loop
X, Y = get_batch('train') 
t0 = time.time()
local_iter_num = 0 
raw_model = model.module if ddp else model 
running_mfu = -1.0

while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"Saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    
    if iter_num == 0 and eval_only:
        break

    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps 
        X, Y = get_batch('train')
        scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
    
    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()