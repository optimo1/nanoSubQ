import os
import time
import math
from contextlib import nullcontext

import numpy as np
import torch

from model import nanoSubQConfig, nanoSubQ

# -----------------------------------------------------------------------------
# Configuration
out_dir = 'out'
eval_interval = 250
log_interval = 10
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'

# Data
dataset = 'nanoSubQ'
gradient_accumulation_steps = 8
batch_size = 4
block_size = 1024

# Model Hyperparameters
num_layers = 12
num_q_heads = 12
num_kv_heads = 4
d_model = 768
dropout = 0.0
bias = False        # Disable router biases to maintain logit balance
clamp_val = 4.0     # Logit clamping threshold

# Optimizer Parameters
learning_rate = 3e-4 # STE-tuned learning rate
max_iters = 600000
weight_decay = 0.1   # Excludes router params automatically in configure_optimizers
beta1 = 0.9
beta2 = 0.95
grad_clip = 0.5      # Safety guardrail to block STE gradient explosions

# Learning Rate Schedule
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 3e-5

# Temperature Schedule (STE Router)
temp_start = 2.0
temp_end = 0.1

# System
device = 'cuda'
dtype = 'float16'
compile = False

# Configuration Overrides
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) if os.path.exists('configurator.py') else None
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
print(f"Tokens per iteration will be: {tokens_per_iter:,}")

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Data Loader
data_dir = os.path.join('data', dataset)
def get_batch(split):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# Model Initialization
iter_num = 0
best_val_loss = 1e9

model_args = dict(
    num_layers=num_layers,
    num_q_heads=num_q_heads,
    num_kv_heads=num_kv_heads,
    d_model=d_model,
    max_seq_len=block_size,
    vocab_size=50304,
    dropout=dropout,
    bias=bias,
    clamp_val=clamp_val
)

if init_from == 'scratch':
    print("Initializing a new nanoSubQ model from scratch...")
    gptconf = nanoSubQConfig(**model_args)
    model = nanoSubQ(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['num_layers', 'num_q_heads', 'num_kv_heads', 'd_model', 'max_seq_len', 'vocab_size', 'dropout', 'bias', 'clamp_val']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = nanoSubQConfig(**model_args)
    model = nanoSubQ(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

model.to(device)

scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None

if compile:
    print("Compiling model...")
    unoptimized_model = model
    model = torch.compile(model)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss, _ = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

def get_temperature(it):
    """ Exponential temperature decay schedule: 2.0 -> 0.1 """
    decay_rate = math.exp(-1.0 * it / (max_iters * 0.3))
    return temp_end + (temp_start - temp_end) * decay_rate

# Training Loop
X, Y = get_batch('train')
t0 = time.time()
raw_model = model

while True:
    # Update Learning Rate
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Exponential Temperature Decay for STE Routers
    current_temp = get_temperature(iter_num)
    raw_model.set_temperature(current_temp)

    # Evaluation Step
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, temp {current_temp:.3f}")
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
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # Forward & Backward Pass with Gradient Accumulation
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss, entropy = model(X, Y)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch('train')
        scaler.scale(loss).backward()

    # Apply Safety Guardrail: Clip Gradients at 0.5
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        lossf = loss.item() * gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.4f}, entropy {entropy.item():.4f}, temp {current_temp:.3f}, time {dt*1000:.2f}ms")
    iter_num += 1

    if iter_num > max_iters:
        break