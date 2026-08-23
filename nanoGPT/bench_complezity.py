import time
import torch
from model import nanoSubQ, nanoSubQConfig

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device != 'cuda':
    print("Warning: CUDA is not available. Memory profiling requires a GPU.")

# Sequence lengths to test
sequence_lengths = [512, 1024, 2048]
batch_size = 4
num_warmup = 5
num_bench = 20

print("=" * 65)
print(f"{'Seq Length (N)':<15} | {'Step Time (ms)':<15} | {'Peak VRAM (MB)':<15}")
print("=" * 65)

for seq_len in sequence_lengths:
    # 1. Initialize Config & Model
    config = nanoSubQConfig(
        vocab_size=50304,
        max_seq_len=seq_len,
        d_model=256,
        num_layers=4,
        num_q_heads=4,
        num_kv_heads=2,
        window_size=8,
    )
    model = nanoSubQ(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # 2. Generate Input Batch
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    # 3. Reset Memory Stats & Warmup GPU
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    model.train()
    for _ in range(num_warmup):
        optimizer.zero_grad(set_to_none=True)
        _, loss, _ = model(x, targets=y)
        loss.backward()
        optimizer.step()
    
    if device == 'cuda':
        torch.cuda.synchronize()

    # 4. Measure Execution Time
    t0 = time.perf_counter()
    for _ in range(num_bench):
        optimizer.zero_grad(set_to_none=True)
        _, loss, _ = model(x, targets=y)
        loss.backward()
        optimizer.step()
        
    if device == 'cuda':
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    avg_step_ms = ((t1 - t0) / num_bench) * 1000.0
    
    # 5. Measure Peak Allocated Memory
    if device == 'cuda':
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        peak_vram_mb = 0.0

    print(f"{seq_len:<15} | {avg_step_ms:<15.2f} | {peak_vram_mb:<15.2f}")

print("=" * 65)