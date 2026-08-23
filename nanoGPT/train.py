import os
import time
import math
import gc
import numpy as np
import torch
import torch.nn as nn
from model import nanoSubQ, nanoSubQConfig, TemperatureScheduler

# -----------------------------------------------------------------------------
# Config & Memory Hyperparameters
# -----------------------------------------------------------------------------
data_dir = 'data'
out_dir = 'out'

# Reduced physical batch size to save VRAM, with accumulation to maintain gradient quality
micro_batch_size = 16   # Dropped from 32 -> 16 to fit safely in GPU VRAM
gradient_accumulation_steps = 2 # Effective batch size = 16 * 2 = 32
block_size = 512     

tokens_per_iter = micro_batch_size * block_size * gradient_accumulation_steps # 16,384 tokens / iter

# Load binary dataset mappings
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

# Calculate exact iterations required for 1 full epoch
tokens_in_train = len(train_data)
max_iters = tokens_in_train // tokens_per_iter

print(f"Total tokens in train.bin: {tokens_in_train:,}")
print(f"Effective batch size in tokens: {tokens_per_iter:,}")
print(f"Calculated iterations for 1 full epoch: {max_iters:,} steps")

eval_interval = max(200, max_iters // 20)  # Log ~20 validation checkpoints across run
log_interval = 20
eval_iters = 50

# Learning Rate & Optimizer Config
learning_rate = 6e-4 
min_lr = 6e-5        
weight_decay = 0.1
warmup_iters = min(2000, int(max_iters * 0.05))
max_grad_norm = 1.0  

# Temperature Schedule Config (Exponential Decay: 2.0 -> 0.1)
t_max = 2.0
t_min = 0.1
t_decay_rate = 3.0

# Model Config (~15M Parameters)
config = nanoSubQConfig(
    vocab_size=50304,
    max_seq_len=block_size,
    d_model=384,        
    num_layers=6,       
    num_q_heads=6,      
    num_kv_heads=2,      
    window_size=8,
    dropout=0.0,
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ptdtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
scaler = torch.amp.GradScaler('cuda', enabled=(ptdtype == torch.float16))

print(f"Using device: {device} | Precision: {ptdtype}")
os.makedirs(out_dir, exist_ok=True)

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (micro_batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

# -----------------------------------------------------------------------------
# Init Model & Optimizer
# -----------------------------------------------------------------------------
model = nanoSubQ(config).to(device)

def configure_optimizer(model, weight_decay, learning_rate):
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if param.ndim < 2 or "router" in name or "temperature" in name or "clamp" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95))

optimizer = configure_optimizer(model, weight_decay, learning_rate)

# Temperature Scheduler synchronized across full max_iters
temp_scheduler = TemperatureScheduler(model, t_max=t_max, t_min=t_min, total_steps=max_iters, decay_rate=t_decay_rate)

# Cosine Learning Rate Schedule
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------
model.train()
t0 = time.time()

for iter_num in range(1, max_iters + 1):
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    temp = temp_scheduler.step(iter_num)

    optimizer.zero_grad(set_to_none=True)
    accum_loss = 0.0

    # Gradient Accumulation Loop
    for micro_step in range(gradient_accumulation_steps):
        x, y = get_batch('train')
        
        with torch.amp.autocast(device_type='cuda', dtype=ptdtype):
            logits, loss, entropy = model(x, targets=y)
            loss = loss / gradient_accumulation_steps  # Scale loss for accumulation

        accum_loss += loss.item() * gradient_accumulation_steps
        
        if ptdtype == torch.float16:
            scaler.scale(loss).backward()
        else:
            loss.backward()

    # Step Optimizer
    if ptdtype == torch.float16:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
    
    if iter_num % log_interval == 0:
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        print(f"step {iter_num:6d}/{max_iters} | loss {accum_loss:.4f} | entropy {entropy.item():.4f} | lr {lr:.6f} | temp {temp:.2f} | time {dt*1000/log_interval:.2f}ms/step")

    if iter_num % eval_interval == 0 or iter_num == max_iters:
        model.eval()
        with torch.no_grad():
            val_losses = []
            for _ in range(eval_iters):
                x_v, y_v = get_batch('val')
                with torch.amp.autocast(device_type='cuda', dtype=ptdtype):
                    _, v_loss, _ = model(x_v, targets=y_v)
                val_losses.append(v_loss.item())
            mean_val_loss = sum(val_losses) / len(val_losses)
            print(f"\n--- EVAL @ Step {iter_num}/{max_iters} | Validation Loss: {mean_val_loss:.4f} ---\n")
            
            ckpt_path = os.path.join(out_dir, f'ckpt_step_{iter_num}.pt')
            torch.save(model.state_dict(), ckpt_path)
            
        # Clean up VRAM after eval
        torch.cuda.empty_cache()
        gc.collect()
        model.train()

print("Full dataset training completed successfully.")