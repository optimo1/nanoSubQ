import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
from model import nanoSubQ, nanoSubQConfig

# -----------------------------------------------------------------------------
# Config & Hyperparameters
# -----------------------------------------------------------------------------
data_dir = 'data'
out_dir = 'out'
batch_size = 8
block_size = 128  # max_seq_len
max_iters = 1000
eval_interval = 100
log_interval = 10
eval_iters = 20

# Learning Rate & Optimizer Config
learning_rate = 3e-4
min_lr = 3e-5       # 10% of peak learning rate for cosine decay
weight_decay = 0.1
warmup_iters = 100
max_grad_norm = 0.5 # Safety Guardrail: Block STE temperature decay gradient explosions

# Temperature Schedule Config (Exponential Decay: 2.0 -> 0.1)
t_max = 2.0
t_min = 0.1
t_decay_rate = 3.0

# Model Config
config = nanoSubQConfig(
    vocab_size=50304,
    max_seq_len=block_size,
    d_model=128,
    num_layers=4,
    num_q_heads=4,
    num_kv_heads=2,
    window_size=8,
)

device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")
os.makedirs(out_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# Data Loader (np.memmap)
# -----------------------------------------------------------------------------
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

# -----------------------------------------------------------------------------
# Exponential Temperature Scheduler
# -----------------------------------------------------------------------------
class TemperatureScheduler:
    def __init__(self, model: nn.Module, t_max: float = 2.0, t_min: float = 0.1, total_steps: int = 1000, decay_rate: float = 3.0):
        self.model = model
        self.t_max = t_max
        self.t_min = t_min
        self.total_steps = total_steps
        self.decay_rate = decay_rate

    def step(self, current_step: int) -> float:
        progress = current_step / max(1, self.total_steps)
        # Exponential temperature decay formula
        temp = self.t_min + (self.t_max - self.t_min) * math.exp(-self.decay_rate * progress)
        
        # Apply updated temperature across all router modules in nanoSubQ
        for module in self.model.modules():
            if hasattr(module, 'set_temperature'):
                module.set_temperature(temp)
        return temp

# -----------------------------------------------------------------------------
# Init Model & Parameter Grouping (Selective Weight Decay)
# -----------------------------------------------------------------------------
model = nanoSubQ(config).to(device)

def configure_optimizer(model, weight_decay, learning_rate):
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Exclude 1D tensors (biases, LayerNorm) and router control/temperature/clamp parameters
        if param.ndim < 2 or "router" in name or "temperature" in name or "clamp" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95))
    return optimizer

optimizer = configure_optimizer(model, weight_decay, learning_rate)
temp_scheduler = TemperatureScheduler(model, t_max=t_max, t_min=t_min, total_steps=max_iters, decay_rate=t_decay_rate)

# Cosine Learning Rate Schedule with Warmup
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
    # 1. Step Learning Rate (Cosine Decay) & Temperature (Exponential Decay)
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    temp = temp_scheduler.step(iter_num)

    # 2. Forward Pass
    x, y = get_batch('train')
    logits, loss, entropy = model(x, targets=y)
    
    # 3. Backward Pass & Safety Guardrail: Gradient Clipping (0.5)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    optimizer.step()
    
    # 4. Logging & Binary Entropy Tracking
    if iter_num % log_interval == 0:
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        print(f"step {iter_num:4d} | loss {loss.item():.4f} | entropy {entropy.item():.4f} | lr {lr:.6f} | temp {temp:.2f} | time {dt*1000/log_interval:.2f}ms/step")

    # 5. Evaluation Loop & Checkpointing
    if iter_num % eval_interval == 0 or iter_num == max_iters:
        model.eval()
        with torch.no_grad():
            val_losses = []
            for _ in range(eval_iters):
                x_v, y_v = get_batch('val')
                _, v_loss, _ = model(x_v, targets=y_v)
                val_losses.append(v_loss.item())
            mean_val_loss = sum(val_losses) / len(val_losses)
            print(f"\n--- EVAL @ Step {iter_num} | Validation Loss: {mean_val_loss:.4f} ---\n")
            
            ckpt_path = os.path.join(out_dir, f'ckpt_step_{iter_num}.pt')
            torch.save(model.state_dict(), ckpt_path)
        model.train()

print("Training finished successfully.")