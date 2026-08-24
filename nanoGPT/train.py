import os
import time
import math
import gc
import numpy as np
import torch
import torch.nn as nn
from model import nanoSubQ, nanoSubQConfig, TemperatureScheduler

data_dir = 'data'
out_dir = 'out'

micro_batch_size = 16
gradient_accumulation_steps = 2
block_size = 512

train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

max_iters = 50000
eval_interval = 2500
log_interval = 20
eval_iters = 50
warmup_iters = 1000

learning_rate = 5.0e-4   
min_lr = 5.0e-5          
weight_decay = 0.1
max_grad_norm = 0.5      

entropy_coef = 0.0
usage_coef = 0.01

t_max = 2.0
t_min = 0.1

config = nanoSubQConfig(
    vocab_size=50304,
    max_seq_len=1024,
    d_model=384,
    num_layers=6,
    num_q_heads=6,
    num_kv_heads=2,
    window_size=8,
    dropout=0.0,
    router_lr_mult=0.5,     
    usage_target=0.5,
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

raw_model = nanoSubQ(config)

def configure_optimizer(model, weight_decay, learning_rate, router_lr_mult):
    decay_params = []
    no_decay_params = []
    router_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "router" in name:
            router_params.append(param)
        elif param.ndim < 2 or "temperature" in name or "ln" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay, "lr": learning_rate},
        {"params": no_decay_params, "weight_decay": 0.0, "lr": learning_rate},
        {"params": router_params, "weight_decay": 0.0, "lr": learning_rate * router_lr_mult},
    ]

    return torch.optim.AdamW(optim_groups, betas=(0.9, 0.95))

optimizer = configure_optimizer(raw_model, weight_decay, learning_rate, config.router_lr_mult)
temp_scheduler = TemperatureScheduler(
    raw_model, t_max=t_max, t_min=t_min, total_steps=max_iters
)

if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    print(f"Enabling DataParallel across {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(raw_model).to(device)
else:
    model = raw_model.to(device)

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

for iter_num in range(1, max_iters + 1):
    lr = get_lr(iter_num)

    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = lr
    optimizer.param_groups[2]['lr'] = lr * config.router_lr_mult

    # Unpack the single returned temp variable cleanly
    temp = temp_scheduler.step(iter_num)
    if isinstance(temp, tuple):
        temp = temp[0]

    optimizer.zero_grad(set_to_none=True)
    accum_loss = 0.0
    accum_entropy = 0.0
    accum_usage = 0.0

    for micro_step in range(gradient_accumulation_steps):
        x, y = get_batch('train')

        with torch.amp.autocast(device_type='cuda', dtype=ptdtype):
            logits, ce_loss, entropy, usage_penalty = model(x, targets=y)
            total_loss = ce_loss.mean() + usage_coef * usage_penalty.mean()
            if entropy_coef > 0.0:
                total_loss = total_loss + entropy_coef * entropy.mean()
            loss_scaled = total_loss / gradient_accumulation_steps

        accum_loss += ce_loss.mean().item()
        accum_entropy += entropy.mean().item()
        accum_usage += usage_penalty.mean().item()

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
        avg_entropy = accum_entropy / gradient_accumulation_steps
        avg_usage = accum_usage / gradient_accumulation_steps
        print(f"step {iter_num:5d}/{max_iters} | loss {avg_loss:.4f} | entropy {avg_entropy:.4f} | "
              f"usage_pen {avg_usage:.4f} | lr {lr:.6f} | temp {temp:.2f} | "
              f"time {dt*1000/log_interval:.2f}ms/step")

    if iter_num % eval_interval == 0 or iter_num == max_iters:
        model.eval()
        with torch.no_grad():
            val_losses = []
            for _ in range(eval_iters):
                x_v, y_v = get_batch('val')
                with torch.amp.autocast(device_type='cuda', dtype=ptdtype):
                    _, v_loss, _, _ = model(x_v, targets=y_v)
                val_losses.append(v_loss.mean().item())
            mean_val_loss = sum(val_losses) / len(val_losses)
            print(f"\n--- EVAL @ Step {iter_num}/{max_iters} | Validation Loss: {mean_val_loss:.4f} ---\n")

            ckpt_path = os.path.join(out_dir, f'ckpt_step_{iter_num}.pt')
            torch.save(raw_model.state_dict(), ckpt_path)

        torch.cuda.empty_cache()
        gc.collect()
        model.train()

print("Training finished successfully!")