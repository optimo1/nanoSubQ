import time
import math
import torch
from model import nanoSubQ, nanoSubQConfig

# -----------------------------------------------------------------------------
# Configuration: Single Batch (64 Tokens)
# -----------------------------------------------------------------------------
batch_size = 1
block_size = 64
max_iters = 100
learning_rate = 1e-3  # Slightly higher LR for quick convergence test

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
print(f"Running Overfitting Sanity Check on device: {device}\n")

model = nanoSubQ(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Lock down a SINGLE dummy batch of 64 tokens
torch.manual_seed(42)
x_fixed = torch.randint(0, config.vocab_size, (batch_size, block_size), device=device)
y_fixed = torch.randint(0, config.vocab_size, (batch_size, block_size), device=device)

# Exponential Temperature Decay Helper
def get_temp(step, t_max=2.0, t_min=0.1, total_steps=100, decay_rate=3.0):
    progress = step / total_steps
    return t_min + (t_max - t_min) * math.exp(-decay_rate * progress)

# -----------------------------------------------------------------------------
# Overfitting Test Loop
# -----------------------------------------------------------------------------
model.train()
initial_loss = None
final_loss = None

t0 = time.time()

for step in range(1, max_iters + 1):
    # Update temperature across router modules
    temp = get_temp(step)
    for module in model.modules():
        if hasattr(module, 'set_temperature'):
            module.set_temperature(temp)

    # Forward & Backward Pass
    optimizer.zero_grad(set_to_none=True)
    logits, loss, entropy = model(x_fixed, targets=y_fixed)
    
    loss.backward()
    
    # Safety Guardrail: Clip Gradients (0.5)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    optimizer.step()

    if step == 1:
        initial_loss = loss.item()
        
    if step % 10 == 0 or step == max_iters:
        print(f"Step {step:3d} | Loss: {loss.item():.4f} | Entropy: {entropy.item():.4f} | Temp: {temp:.2f}")

    final_loss = loss.item()

t1 = time.time()

# -----------------------------------------------------------------------------
# Results Verification
# -----------------------------------------------------------------------------
print("\n" + "=" * 50)
print(f"Initial Loss (Step 1): {initial_loss:.4f}")
print(f"Final Loss (Step 100): {final_loss:.4f}")
print(f"Execution Time:       {(t1 - t0):.2f}s")
print("=" * 50)

if final_loss < 0.1:
    print("SUCCESS: Model overfit the single batch. STE gradients & guardrails working properly!")
elif final_loss < initial_loss * 0.5:
    print("WARNING: Loss decreased significantly but did not drop near 0.0. Consider running for 200 steps.")
else:
    print("FAILURE: Loss failed to collapse. Check router backpropagation / STE gradient flow.")