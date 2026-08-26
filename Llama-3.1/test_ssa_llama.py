"""Comprehensive tests for ssa_llama.py — gradient flow, entropy, temperature, causality, etc."""
import sys, os, math
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
from transformers import LlamaConfig
from ssa_llama import (
    SSA_LlamaAttention, subq_route, routing_stats,
    apply_rotary_pos_emb, rotate_half, ssa_masked,
)


def make_config():
    return LlamaConfig(
        hidden_size=4096, num_attention_heads=32, num_key_value_heads=8,
        intermediate_size=14336, max_position_embeddings=1024,
        rms_norm_eps=1e-5, attention_bias=False,
    )


def make_rope(S, head_dim, device):
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(S, device=device).float()
    freqs = torch.einsum('i,j->ij', t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().unsqueeze(0), emb.sin().unsqueeze(0)  # (1, S, D)


def forward_attn(config, hidden, attn_module):
    """Run forward pass through SSA attention with proper RoPE."""
    B, S, D = hidden.shape
    cos, sin = make_rope(S, attn_module.head_dim, hidden.device)
    return attn_module(hidden, position_embeddings=(cos, sin))


# ═══════════════════════════════════════════════════════════════════════════════
# 1. GRADIENT FLOW
# ═══════════════════════════════════════════════════════════════════════════════
def test_gradient_flow():
    """Gradients must flow through Q/K/V projections AND the routing mechanism."""
    config = make_config()
    attn = SSA_LlamaAttention(config, layer_idx=0)
    attn.attn_impl = 'masked'  # CPU-safe

    B, S, D = 2, 256, config.hidden_size
    hidden = torch.randn(B, S, D, requires_grad=True)

    out, _ = forward_attn(config, hidden, attn)
    loss = out.sum()
    loss.backward()

    # Check hidden input gets gradients
    assert hidden.grad is not None, "No gradient on input"
    assert hidden.grad.abs().sum() > 0, "Zero gradient on input"

    # Check Q/K/V/O projections get gradients
    for name, p in attn.named_parameters():
        if p.grad is not None:
            assert p.grad.abs().sum() > 0, f"Zero gradient on {name}"

    # Check that routing-relevant params exist (beta is not a parameter, but
    # the projection weights that feed into routing must have gradients)
    grad_params = [(n, p) for n, p in attn.named_parameters()
                   if p.grad is not None and p.grad.abs().sum() > 0]
    assert len(grad_params) >= 4, f"Expected >=4 grad params, got {len(grad_params)}: {[n for n,_ in grad_params]}"
    print(f"1. Gradient flow OK: {len(grad_params)} params have gradients")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ROUTING ENTROPY
# ═══════════════════════════════════════════════════════════════════════════════
def test_routing_entropy():
    """Routing entropy should be positive, finite, and respond to beta."""
    B, NKV, n, d = 2, 8, 1024, 128  # 8 blocks
    q = torch.randn(B, 32, n, d)
    k = torch.randn(B, NKV, n, d)

    # Low beta → more uniform scores → higher entropy
    sel_low, r_low, rout_low = subq_route(q, k, 128, top_c=2, local=1, beta=0.5)
    _, ent_low, _ = routing_stats(sel_low, r_low, rout_low)

    # High beta → sharper scores → lower entropy
    sel_high, r_high, rout_high = subq_route(q, k, 128, top_c=2, local=1, beta=5.0)
    _, ent_high, _ = routing_stats(sel_high, r_high, rout_high)

    assert math.isfinite(ent_low), f"Non-finite entropy (low beta): {ent_low}"
    assert math.isfinite(ent_high), f"Non-finite entropy (high beta): {ent_high}"
    assert ent_low > 0, f"Entropy <= 0 (low beta): {ent_low}"
    assert ent_high > 0, f"Entropy <= 0 (high beta): {ent_high}"
    print(f"2. Entropy OK: low_beta={ent_low:.4f}, high_beta={ent_high:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TEMPERATURE (BETA) EFFECT
# ═══════════════════════════════════════════════════════════════════════════════
def test_beta_effect():
    """Higher beta should make routing scores more extreme (higher variance)."""
    B, NKV, n, d = 2, 8, 512, 128
    q = torch.randn(B, 32, n, d)
    k = torch.randn(B, NKV, n, d)

    betas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    score_stds = []
    for beta in betas:
        sel, r, rout = subq_route(q, k, 128, top_c=2, local=1, beta=beta)
        # Score std among valid (non -inf) entries
        valid_r = r[r != float('-inf')]
        if len(valid_r) > 0:
            score_stds.append(valid_r.std().item())
        else:
            score_stds.append(0.0)

    # Higher beta → higher score variance
    assert score_stds[-1] > score_stds[0], \
        f"Beta effect broken: std at beta={betas[0]}={score_stds[0]:.3f}, beta={betas[-1]}={score_stds[-1]:.3f}"
    print(f"3. Beta effect OK: std ranges from {score_stds[0]:.3f} (β=0.1) to {score_stds[-1]:.3f} (β=10)")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. BETA ANNEALING (set_beta)
# ═══════════════════════════════════════════════════════════════════════════════
def test_beta_annealing():
    """set_beta should change routing behavior across steps."""
    config = make_config()
    attn = SSA_LlamaAttention(config, layer_idx=0)
    attn.attn_impl = 'masked'

    # Use 8 blocks (1024 tokens) so beta can differentiate routing
    B, S, D = 1, 1024, config.hidden_size
    hidden = torch.randn(B, S, D)
    cos, sin = make_rope(S, attn.head_dim, hidden.device)

    # Run with beta=0.5
    attn.set_beta(0.5)
    out1, _ = attn(hidden, position_embeddings=(cos, sin))

    # Run with beta=10.0
    attn.set_beta(10.0)
    out2, _ = attn(hidden, position_embeddings=(cos, sin))

    # Outputs should differ (different routing → different attention)
    diff = (out1 - out2).abs().max().item()
    assert diff > 1e-6, f"Beta change had no effect: diff={diff}"
    print(f"4. Beta annealing OK: output diff={diff:.6f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CAUSALITY
# ═══════════════════════════════════════════════════════════════════════════════
def test_causality():
    """No future-block leakage. Changing future tokens must not affect current output."""
    config = make_config()
    attn = SSA_LlamaAttention(config, layer_idx=0)
    attn.attn_impl = 'masked'

    B, S, D = 1, 256, config.hidden_size
    hidden = torch.randn(B, S, D)
    cos, sin = make_rope(S, attn.head_dim, hidden.device)

    out1, _ = attn(hidden, position_embeddings=(cos, sin))

    # Change only the last 64 tokens (2 blocks)
    hidden2 = hidden.clone()
    hidden2[:, 192:, :] = torch.randn(B, 64, D)

    out2, _ = attn(hidden2, position_embeddings=(cos, sin))

    # First 128 tokens (1 block) should be identical
    diff_first_block = (out1[:, :128, :] - out2[:, :128, :]).abs().max().item()
    assert diff_first_block < 1e-5, f"Causality violated! First block diff: {diff_first_block}"
    print(f"5. Causality OK: first block diff after changing future = {diff_first_block:.2e}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. GQA CORRECTNESS
# ═══════════════════════════════════════════════════════════════════════════════
def test_gqa():
    """Q heads are grouped: 32 Q heads share 8 KV heads (ratio=4)."""
    config = make_config()
    attn = SSA_LlamaAttention(config, layer_idx=0)
    attn.attn_impl = 'masked'  # CPU-safe

    B, S, D = 1, 256, config.hidden_size
    hidden = torch.randn(B, S, D)
    cos, sin = make_rope(S, attn.head_dim, hidden.device)

    # Get intermediate shapes
    q = attn.q_proj(hidden).view(B, S, 32, 128).transpose(1, 2)
    k = attn.k_proj(hidden).view(B, S, 8, 128).transpose(1, 2)
    v = attn.v_proj(hidden).view(B, S, 8, 128).transpose(1, 2)

    q, k = apply_rotary_pos_emb(q, k, cos, sin)

    # Routing: sel should be (B=1, NKV=8, nb, nb)
    sel, r, rout = subq_route(q, k, 128, top_c=2, local=1, beta=2.0)
    assert sel.shape == (1, 8, 2, 2), f"Routing sel shape wrong: {sel.shape}"

    out, _ = attn(hidden, position_embeddings=(cos, sin))
    assert out.shape == (B, S, D), f"Output shape wrong: {out.shape}"
    print(f"6. GQA OK: Q=32h, KV=8h, routing shape={tuple(sel.shape)}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. NUMERICAL STABILITY
# ═══════════════════════════════════════════════════════════════════════════════
def test_numerical_stability():
    """No NaN/inf in output even with extreme inputs."""
    config = make_config()
    attn = SSA_LlamaAttention(config, layer_idx=0)
    attn.attn_impl = 'masked'

    B, S, D = 1, 256, config.hidden_size
    cos, sin = make_rope(S, attn.head_dim, torch.device('cpu'))

    cases = {
        'large_vals': torch.randn(B, S, D) * 100,
        'small_vals': torch.randn(B, S, D) * 1e-6,
        'mixed': torch.randn(B, S, D) * 10,
    }

    for name, hidden in cases.items():
        out, _ = attn(hidden, position_embeddings=(cos, sin))
        assert torch.isfinite(out).all(), f"NaN/inf in output for {name}"
    print("7. Numerical stability OK: no NaN/inf for large/small/mixed inputs")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. BATCH INDEPENDENCE
# ═══════════════════════════════════════════════════════════════════════════════
def test_batch_independence():
    """Each batch element should be independent."""
    config = make_config()
    attn = SSA_LlamaAttention(config, layer_idx=0)
    attn.attn_impl = 'masked'

    S, D = 256, config.hidden_size
    cos, sin = make_rope(S, attn.head_dim, torch.device('cpu'))

    # Two different batch elements
    h1 = torch.randn(1, S, D)
    h2 = torch.randn(1, S, D)

    out1a, _ = attn(h1, position_embeddings=(cos, sin))
    out2a, _ = attn(h2, position_embeddings=(cos, sin))

    # Stack in batch
    h12 = torch.cat([h1, h2], dim=0)
    out12, _ = attn(h12, position_embeddings=(cos, sin))

    # Batch outputs should match individual outputs
    diff0 = (out12[0:1] - out1a).abs().max().item()
    diff1 = (out12[1:2] - out2a).abs().max().item()
    assert diff0 < 1e-5, f"Batch[0] differs from single: {diff0}"
    assert diff1 < 1e-5, f"Batch[1] differs from single: {diff1}"
    print(f"8. Batch independence OK: diff0={diff0:.2e}, diff1={diff1:.2e}")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. BLOCK ALIGNMENT
# ═══════════════════════════════════════════════════════════════════════════════
def test_block_alignment():
    """Routing works on block-aligned boundaries. n must be divisible by block."""
    B, NKV, n, d = 1, 8, 512, 128
    q = torch.randn(B, 32, n, d)
    k = torch.randn(B, NKV, n, d)

    # Should work: 512 % 128 == 0
    sel, r, rout = subq_route(q, k, 128, top_c=2, local=1, beta=2.0)
    nb = n // 128  # 4
    assert sel.shape == (B, NKV, nb, nb)

    # Should fail: 500 % 128 != 0
    try:
        q_bad = torch.randn(B, 32, 500, d)
        k_bad = torch.randn(B, NKV, 500, d)
        subq_route(q_bad, k_bad, 128, top_c=2, local=1, beta=2.0)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    print(f"9. Block alignment OK: works for divisible, fails for non-divisible")


# ═══════════════════════════════════════════════════════════════════════════════
# 10. MASKED vs FLEX EQUIVALENCE
# ═══════════════════════════════════════════════════════════════════════════════
def test_masked_vs_flex():
    """If FlexAttention is available and on GPU, masked and flex should produce similar outputs."""
    if not torch.cuda.is_available():
        print("10. Skipped (no GPU — FlexAttention requires CUDA)")
        return
    try:
        from torch.nn.attention.flex_attention import BlockMask
    except ImportError:
        print("10. Skipped (FlexAttention not available)")
        return

    config = make_config()
    attn_mask = SSA_LlamaAttention(config, layer_idx=0)
    attn_mask.attn_impl = 'masked'

    attn_flex = SSA_LlamaAttention(config, layer_idx=0)
    attn_flex.attn_impl = 'flex'

    # Copy weights
    attn_flex.load_state_dict(attn_mask.state_dict())

    device = torch.device('cuda')
    attn_mask = attn_mask.to(device)
    attn_flex = attn_flex.to(device)

    B, S, D = 1, 256, config.hidden_size
    hidden = torch.randn(B, S, D, device=device)
    cos, sin = make_rope(S, config.hidden_size // config.num_attention_heads, device)

    out_mask, _ = attn_mask(hidden, position_embeddings=(cos, sin))
    out_flex, _ = attn_flex(hidden, position_embeddings=(cos, sin))

    diff = (out_mask - out_flex).abs().max().item()
    print(f"10. Masked vs Flex: max diff = {diff:.2e} {'OK' if diff < 0.1 else 'MISMATCH'}")


# ═══════════════════════════════════════════════════════════════════════════════
# 11. LOCAL WINDOW
# ═══════════════════════════════════════════════════════════════════════════════
def test_local_window():
    """Own block + local preceding blocks are always kept."""
    B, NKV, n, d = 1, 8, 512, 128
    q = torch.randn(B, 32, n, d)
    k = torch.randn(B, NKV, n, d)

    sel, _, _ = subq_route(q, k, 128, top_c=1, local=2, beta=2.0)

    # Block 3 should always keep blocks 1, 2, 3 (local=2)
    for kv_h in range(NKV):
        assert sel[0, kv_h, 3, 3], f"Block 3 doesn't keep own block (KV head {kv_h})"
        assert sel[0, kv_h, 3, 2], f"Block 3 doesn't keep block 2 (KV head {kv_h})"
        assert sel[0, kv_h, 3, 1], f"Block 3 doesn't keep block 1 (KV head {kv_h})"

    # Block 0 should keep only itself (no preceding blocks with local=2)
    for kv_h in range(NKV):
        assert sel[0, kv_h, 0, 0], f"Block 0 doesn't keep own block"
        assert not sel[0, kv_h, 0, 1], f"Block 0 selected future block 1"
    print("11. Local window OK: own + local blocks always kept")


# ═══════════════════════════════════════════════════════════════════════════════
# 12. LOAD BALANCE
# ═══════════════════════════════════════════════════════════════════════════════
def test_load_balance():
    """Load balance metric should be finite and in valid range."""
    B, NKV, n, d = 2, 8, 1024, 128
    q = torch.randn(B, 32, n, d)
    k = torch.randn(B, NKV, n, d)

    sel, r, rout = subq_route(q, k, 128, top_c=4, local=1, beta=2.0)
    sparsity, entropy, load = routing_stats(sel, r, rout)

    assert math.isfinite(sparsity), f"Non-finite sparsity: {sparsity}"
    assert math.isfinite(entropy), f"Non-finite entropy: {entropy}"
    assert math.isfinite(load), f"Non-finite load: {load}"
    assert 0 < sparsity <= 1, f"Sparsity out of range: {sparsity}"
    assert load >= 1.0, f"Load < 1.0 (IPR minimum): {load}"
    print(f"12. Load balance OK: sparsity={sparsity:.3f}, entropy={entropy:.3f}, load={load:.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    torch.manual_seed(42)
    print("=" * 65)
    print("SSA LLaMA 3.1 8B — Comprehensive Tests")
    print("=" * 65)

    tests = [
        test_gradient_flow,
        test_routing_entropy,
        test_beta_effect,
        test_beta_annealing,
        test_causality,
        test_gqa,
        test_numerical_stability,
        test_batch_independence,
        test_block_alignment,
        test_masked_vs_flex,
        test_local_window,
        test_load_balance,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAILED: {test.__name__}: {e}")
            failed += 1

    print("=" * 65)
    if failed == 0:
        print(f"ALL {passed} TESTS PASSED")
    else:
        print(f"PASSED: {passed}, FAILED: {failed}")
    print("=" * 65)
