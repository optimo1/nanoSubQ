"""Masked (O(n^2)) vs flex (O(n*kappa)) must agree on the same per-query-block routing."""
import os, sys, math
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import nanoSubQ, nanoSubQConfig

torch.manual_seed(0)
cfg = nanoSubQConfig(vocab_size=50304, max_seq_len=256, d_model=128, num_layers=2,
                     num_q_heads=4, num_kv_heads=2, block=128, top_c=4, local=1)
idx = torch.randint(0, cfg.vocab_size, (4, 256))
tgt = torch.randint(0, cfg.vocab_size, (4, 256))

m = nanoSubQ(cfg).train()                      # ONE model; toggle attn_impl on its layers

def set_impl(model, impl):
    for mod in model.modules():
        if hasattr(mod, 'attn_impl'):
            mod.attn_impl = impl

outs = {}
for impl in ('masked', 'flex'):
    set_impl(m, impl)
    m.zero_grad(set_to_none=True)
    logits, loss, sp, ent, ld = m(idx, tgt)
    loss.sum().backward()
    g = sum(p.grad.abs().sum().item() for p in m.parameters() if p.grad is not None)
    assert g > 0, "no gradient flowed"
    for v in (loss.item(), sp.item(), ent.item(), ld.item()):
        assert math.isfinite(v), "non-finite diagnostic"
    outs[impl] = logits.detach()
    print(f"{impl:6s}: loss {loss.item():.3f} sparsity {sp.item():.2f} entropy {ent.item():.3f} load {ld.item():.2f}")

diff = (outs['masked'] - outs['flex']).abs().max().item()
print(f"masked vs flex logits max diff: {diff:.2e}")
assert diff < 1e-3, f"flex != masked (diff {diff})"
print("EQUIV TEST PASSED")
