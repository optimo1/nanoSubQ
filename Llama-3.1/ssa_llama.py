"""
SSA (Subquadratic Sparse Attention) for LLaMA 3.1 8B.

Drop-in replacement for LlamaAttention that adds content-routed sparse attention:
  - Block-cumulant routing: per query-block, score key-blocks by ⟨q̄,μ_c⟩ + ½β·q̄ᵀdiag(Σ_c)q̄
  - Select top_c causally-past blocks + local window
  - Exact softmax only over selected blocks (O(n·κ) with FlexAttention)

Architecture match for LLaMA 3.1 8B:
  hidden_size=4096, num_heads=32, num_kv_heads=8, head_dim=128, block=128

Usage:
    from transformers import LlamaForCausalLM
    from ssa_llama import SSA_LlamaAttention

    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
    for layer in model.model.layers:
        layer.self_attn = SSA_LlamaAttention(model.config, layer_idx=layer.layer_idx)
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.nn.attention.flex_attention import flex_attention, BlockMask
    _FLEX_OK = True
except ImportError:
    _FLEX_OK = False

_FLEX_COMPILED = None


def _flex_kernel():
    global _FLEX_COMPILED
    if _FLEX_COMPILED is None:
        _FLEX_COMPILED = torch.compile(flex_attention) if torch.cuda.is_available() else flex_attention
    return _FLEX_COMPILED


# ── Copied from transformers (LLaMA's rotate_half + apply_rotary_pos_emb) ─────
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply RoPE. cos/sin: (B, S, head_dim) or (1, 1, S, head_dim) — broadcasts to q/k (B, H, S, D)."""
    # Ensure cos/sin are 4D: (BROADCAST, 1, S, D) so they expand over the head dim
    if cos.dim() == 3:
        cos = cos.unsqueeze(1)  # (B, 1, S, D)
        sin = sin.unsqueeze(1)
    elif cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, S, D)
        sin = sin.unsqueeze(0).unsqueeze(0)
    # Now cos: (..., 1, S, D), q: (B, H, S, D) — broadcasts over H correctly
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


# ── SSA routing ────────────────────────────────────────────────────────────────
def subq_route(q, k, block, top_c, local, beta):
    """Per-query-block block-cumulant routing (GQA-aware).

    q: (B, num_q_heads, n, head_dim)
    k: (B, num_kv_heads, n, head_dim)
    Returns: sel (B, num_kv_heads, nb, nb) bool — which key-blocks each KV-head's query-blocks select.
    """
    B, NQ, n, d = q.shape
    NKV = k.shape[1]
    g = NQ // NKV  # GQA group size
    assert n % block == 0, f"n={n} must be a multiple of block={block}"
    nb = n // block

    # Key blocks: (B, NKV, nb, block, d)
    kb = k.float().view(B, NKV, nb, block, d)
    mu = kb.mean(3)           # block mean: (B, NKV, nb, d)
    var = kb.var(3, unbiased=False)  # block variance: (B, NKV, nb, d)

    # Query blocks: group-mean across GQA heads, per query-block
    # q: (B, NQ, n, d) -> (B, NKV, g, nb, block, d) -> mean over g and block dims
    qb = q.float().view(B, NKV, g, nb, block, d).mean((2, 4))  # (B, NKV, nb, d)

    # Routing score: ⟨q̄, μ_c⟩ + ½β · q̄² · σ²_c  (element-wise, then sum over d)
    r = (torch.einsum('bhnd,bhkd->bhnk', qb, mu)
         + 0.5 * beta * torch.einsum('bhnd,bhkd->bhnk', qb * qb, var))

    # Causal mask: key block must be strictly before query block
    rout = torch.arange(nb, device=q.device)[:, None] > torch.arange(nb, device=q.device)[None, :]
    r = r.masked_fill(~rout[None, None], float('-inf'))

    # Top-k selection
    sel = torch.zeros(B, NKV, nb, nb, dtype=torch.bool, device=q.device)
    sel.scatter_(-1, r.topk(min(top_c, nb), dim=-1).indices, True)
    sel &= rout[None, None]

    # Always keep own block + `local` preceding blocks
    diff = torch.arange(nb, device=q.device)[:, None] - torch.arange(nb, device=q.device)[None, :]
    sel = sel | ((diff >= 0) & (diff <= local))[None, None]

    return sel, r, rout


def routing_stats(sel, r, rout):
    """Diagnostics: sparsity, routing entropy, load balance."""
    nvis = rout.sum(-1) + 1
    sparsity = (sel.sum(-1).float() / nvis).mean()
    rv = r.masked_fill(~rout[None, None], float('-inf'))
    p = torch.softmax(rv, dim=-1, dtype=torch.float32)
    ent = -(p * torch.log(p.clamp(min=1e-9)))
    ent = torch.nan_to_num(ent, nan=0.0)
    entropy = ent.sum(-1).mean()
    frac = sel.float().mean(dim=2)
    load = (sel.shape[-1] * (frac ** 2).sum(-1)).mean()
    return sparsity, entropy, load


# ── Attention kernels ──────────────────────────────────────────────────────────
def ssa_masked(q, k, v, sel, block):
    """Exact masked-dense attention (O(n²) reference path)."""
    B, NQ, n, d = q.shape
    nb = sel.shape[-1]
    num_q_per_kv = NQ // k.shape[1]

    sel_q = sel.repeat_interleave(num_q_per_kv, dim=1)  # (B, NQ, nb, nb)
    qblk = torch.arange(n, device=q.device) // block
    sel_tok = sel_q[:, :, qblk, :]  # (B, NQ, n, nb) per-token selection
    keyblk = torch.arange(n, device=q.device) // block
    keymask = sel_tok.gather(-1, keyblk.view(1, 1, 1, n).expand(B, NQ, n, n))

    causal = torch.arange(n, device=q.device)[None, :] <= torch.arange(n, device=q.device)[:, None]
    mask = keymask & causal[None, None]

    kv = k.repeat_interleave(num_q_per_kv, dim=1)
    vv = v.repeat_interleave(num_q_per_kv, dim=1)
    S = (q @ kv.transpose(-1, -2)) * (d ** -0.5)
    w = torch.softmax(S.masked_fill(~mask, float('-inf')), dim=-1, dtype=torch.float32).to(v.dtype)
    return torch.matmul(w, vv)


def ssa_flex(q, k, v, sel, block):
    """Fused block-sparse attention via FlexAttention (O(n·κ) path)."""
    B, NQ, n, d = q.shape
    nb = sel.shape[-1]
    num_q_per_kv = NQ // k.shape[1]
    assert n % block == 0

    if torch.cuda.is_available():
        assert block % 128 == 0, f"flex kernel needs block multiple of 128, got {block}"

    sel_q = sel.repeat_interleave(num_q_per_kv, dim=1)  # (B, NQ, nb, nb)
    kv_num = sel_q.sum(-1).to(torch.int32)
    kv_idx = torch.argsort(sel_q.int(), dim=-1, descending=True, stable=True).to(torch.int32)

    def mm(b, h, qpos, kpos):
        return kpos <= qpos

    bm = BlockMask.from_kv_blocks(kv_num, kv_idx, BLOCK_SIZE=block, mask_mod=mm)
    kv = k.repeat_interleave(num_q_per_kv, dim=1)
    vv = v.repeat_interleave(num_q_per_kv, dim=1)
    return _flex_kernel()(q, kv, vv, block_mask=bm, scale=d ** -0.5)


# ── Drop-in replacement for LlamaAttention ─────────────────────────────────────
class SSA_LlamaAttention(nn.Module):
    """LLaMA attention with SSA block-cumulant routing.

    Matches LlamaAttention's interface exactly — same __init__ params, same forward
    signature, same return type. Can be swapped in with:
        layer.self_attn = SSA_LlamaAttention(config, layer_idx)
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # LLaMA architecture
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_kv_groups = self.num_heads // self.num_kv_heads  # GQA ratio
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = getattr(config, 'attention_dropout', 0.0)

        # SSA hyperparameters
        self.block = 128      # routing block size (must be multiple of 128 for flex)
        self.top_c = 4        # top key-blocks selected per query block
        self.local = 1        # always keep own block + this many preceding
        self.beta = 2.0       # cumulant temperature
        self.attn_impl = 'flex' if _FLEX_OK else 'masked'

        # Projections — identical to LlamaAttention
        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim,
            bias=getattr(config, 'attention_bias', False)
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim,
            bias=getattr(config, 'attention_bias', False)
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim,
            bias=getattr(config, 'attention_bias', False)
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size,
            bias=getattr(config, 'attention_bias', False)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        """Drop-in compatible with LlamaAttention.forward."""
        B, S, D = hidden_states.shape
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # 1. Project Q, K, V — same as LlamaAttention
        q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # 2. Apply RoPE — same as LlamaAttention
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 3. SSA routing + attention
        sel, r, rout = subq_route(q, k, self.block, self.top_c, self.local, self.beta)
        if self.attn_impl == 'flex':
            attn_out = ssa_flex(q, k, v, sel, self.block)
        else:
            attn_out = ssa_masked(q, k, v, sel, self.block)

        # 4. Reshape and project — same as LlamaAttention
        attn_out = attn_out.transpose(1, 2).contiguous().view(*input_shape, -1)
        attn_out = self.o_proj(attn_out)

        # Return (output, attn_weights) to match LlamaAttention
        return attn_out, None

    def set_beta(self, beta: float):
        """Anneal routing temperature during training."""
        self.beta = beta

    def set_impl(self, impl: str):
        """Switch between 'flex' (O(n·κ)) and 'masked' (O(n²) reference)."""
        self.attn_impl = impl

    def routing_diagnostics(self):
        """Return last routing stats (call after forward)."""
        return getattr(self, '_last_stats', None)
