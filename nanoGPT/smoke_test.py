"""Smoke test: (1) data dtype sanity, (2) forward/backward shapes, (3) loss decreases, no NaN."""
import os, sys, math
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import nanoSubQ, nanoSubQConfig

# 1. data dtype check (uint32 on disk)
d = np.memmap(os.path.join(HERE, 'data', 'train.bin'), dtype=np.uint32, mode='r')
first = d[:8].tolist()
assert first[0] == 50257, f"expected <|im_start|> 50257 first, got {first}"
assert not (d[:8][1::2] == 0).all(), f"alternating zeros -> uint16 misread: {first}"
print("1. data OK (uint32):", first)

# 2. tiny model, forward + backward
torch.manual_seed(0)
cfg = nanoSubQConfig(vocab_size=50304, max_seq_len=256, d_model=128, num_layers=2,
                     num_q_heads=4, num_kv_heads=2, block=128, top_c=4, local=1)
m = nanoSubQ(cfg).train()
opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
idx = torch.randint(0, cfg.vocab_size, (4, 256))
tgt = torch.randint(0, cfg.vocab_size, (4, 256))
logits, loss, sparsity, entropy, load = m(idx, tgt)
assert logits.shape == (4, 256, cfg.vocab_size), logits.shape
loss.backward()
g = sum(p.grad.abs().sum().item() for p in m.parameters() if p.grad is not None)
assert g > 0, "no gradient flowed"
for v in (sparsity.item(), entropy.item(), load.item()):
    assert math.isfinite(v), "non-finite diagnostic"
print(f"2. shapes OK; logits {tuple(logits.shape)}; grad_sum {g:.3f}")

# 3. loss decreases over 40 steps on a fixed batch, never NaN
first_loss = None
for step in range(40):
    opt.zero_grad(set_to_none=True)
    logits, loss, sparsity, entropy, load = m(idx, tgt)
    l = loss.item()
    assert math.isfinite(l), f"non-finite loss at step {step}: {l}"
    if step == 0:
        first_loss = l
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
s = sparsity.item()
assert 0.0 < s <= 1.0, f"sparsity {s} out of (0,1]"
print(f"3. loss {first_loss:.3f} -> {l:.3f}; sparsity {s:.3f}")
assert l < first_loss, "loss did not decrease"

# 4. autocast forward: flex_attention must not reject mixed dtypes.
# Regression: fp32 RoPE tables promoted q/k to fp32 under fp16/bf16 autocast while v stayed
# low-precision -> ValueError "Expected query, key, and value to have the same dtype"
# (crashed train.py on the 2xT4/Kaggle box). apply_rotary_emb now casts tables to x.dtype.
with torch.amp.autocast(device_type='cpu', dtype=torch.bfloat16):
    logits_a, loss_a, sp_a, ent_a, ld_a = m(idx, tgt)
assert torch.isfinite(loss_a), "non-finite loss under autocast"
assert torch.isfinite(sp_a) and torch.isfinite(ent_a) and torch.isfinite(ld_a), "non-finite diagnostic under autocast"
print(f"4. autocast forward OK (loss {loss_a.item():.3f}, dtypes uniform)")
print("SMOKE TEST PASSED")
