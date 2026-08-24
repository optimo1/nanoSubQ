import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# --- flex_attention is optional: fall back to masked-dense if unavailable -------------------------
try:
    from torch.nn.attention.flex_attention import flex_attention, BlockMask
    _FLEX_OK = True
except ImportError:
    _FLEX_OK = False
_FLEX_COMPILED = None  # lazy: torch.compile(flex_attention) only on CUDA, first flex use


def _flex_kernel():
    global _FLEX_COMPILED
    if _FLEX_COMPILED is None:
        if torch.cuda.is_available():
            _FLEX_COMPILED = torch.compile(flex_attention)
        else:
            _FLEX_COMPILED = flex_attention
    return _FLEX_COMPILED


@dataclass
class nanoSubQConfig:
    vocab_size: int = 50304
    max_seq_len: int = 1024
    d_model: int = 384
    num_layers: int = 6
    num_q_heads: int = 6
    num_kv_heads: int = 2
    block: int = 128         # routing key-block size. MUST be a multiple of 128: torch's compiled flex
                             #   kernel (>=2.5) requires the mask block size to be a multiple of its
                             #   BLOCK_M/BLOCK_N tiles (128/64 for fp16) or it raises
                             #   "Q and KV block size must be divisible by BLOCK_M and BLOCK_N".
    top_c: int = 4           # top key-blocks selected per query block (causal)
    local: int = 1           # always keep own block + this many preceding blocks
    beta: float = 2.0        # cumulant temperature (repo measured optimum ~2)
    attn_impl: str = 'flex'  # 'flex' = O(n*kappa) fused kernel; 'masked' = O(n^2) exact reference
    label_smoothing: float = 0.1   # soften the CE target so EVERY vocab token gets a gradient
    dropout: float = 0.0


def apply_rotary_emb(x, cos, sin):
    # Cast the fp32 RoPE tables to x's dtype. Otherwise (x * cos) promotes q/k to fp32
    # under fp16/bf16 autocast while v stays low-precision -> flex_attention rejects the
    # mixed q/k/v dtypes with a ValueError.
    d_half = x.shape[-1] // 2
    x1 = x[..., :d_half]
    x2 = x[..., d_half:]
    rotated_x = torch.cat((-x2, x1), dim=-1)
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    return (x * cos) + (rotated_x * sin)


def subq_route(q, k, block, top_c, local, beta):
    """Per-query-block block-cumulant routing, decided per KV head (GQA).
    Requires n % block == 0 (training uses n=512, block=32; the benchmark uses multiples).
    q (B, num_q_heads, n, d); k (B, num_kv_heads, n, d).
    Returns sel (B, num_kv_heads, nb, nb) bool, scores r, rout mask. Queries in a block share the
    selection — the native-sparse design that lets the flex kernel run in O(n*kappa)."""
    B, NQ, n, d = q.shape
    NKV = k.shape[1]
    g = NQ // NKV
    assert n % block == 0, f"n={n} must be a multiple of block={block}"
    nb = n // block
    kb = k.float().view(B, NKV, nb, block, d)
    mu = kb.mean(3)                                        # block mean
    var = kb.var(3, unbiased=False)                        # block spread (2nd cumulant)
    qb = q.float().view(B, NKV, g, nb, block, d).mean((2, 4))   # group-mean q, per query block
    r = torch.einsum('bhnd,bhkd->bhnk', qb, mu) + 0.5 * beta * torch.einsum('bhnd,bhkd->bhnk', qb * qb, var)
    rout = torch.arange(nb, device=q.device)[:, None] > torch.arange(nb, device=q.device)[None, :]  # key block strictly before query block
    r = r.masked_fill(~rout[None, None], float('-inf'))
    sel = torch.zeros(B, NKV, nb, nb, dtype=torch.bool, device=q.device)
    sel.scatter_(-1, r.topk(min(top_c, nb), dim=-1).indices, True)
    sel &= rout[None, None]                                     # topk keeps only strictly-past blocks
    diff = torch.arange(nb, device=q.device)[:, None] - torch.arange(nb, device=q.device)[None, :]
    sel = sel | ((diff >= 0) & (diff <= local))[None, None]     # then always keep own + preceding blocks
    return sel, r, rout


def ssa_masked(q, k, v, sel, block, num_q_per_kv):
    """Exact masked-dense attention over the selected blocks (O(n^2); the reference/fallback)."""
    B, NQ, n, d = q.shape
    nb = sel.shape[-1]
    sel_q = sel.repeat_interleave(num_q_per_kv, dim=1)              # (B,NQ,nb,nb)
    qblk = torch.arange(n, device=q.device) // block
    sel_tok = sel_q[:, :, qblk, :]                                  # (B,NQ,n,nb) per-token selection
    keyblk = torch.arange(n, device=q.device) // block
    keymask = sel_tok.gather(-1, keyblk.view(1, 1, 1, n).expand(B, NQ, n, n))  # (B,NQ,n,n)
    causal = torch.arange(n, device=q.device)[None, :] <= torch.arange(n, device=q.device)[:, None]
    mask = keymask & causal[None, None]
    kv = k.repeat_interleave(num_q_per_kv, dim=1)
    vv = v.repeat_interleave(num_q_per_kv, dim=1)
    S = (q @ kv.transpose(-1, -2)) * (d ** -0.5)
    w = torch.softmax(S.masked_fill(~mask, float('-inf')), dim=-1, dtype=torch.float32).to(v.dtype)
    return torch.matmul(w, vv)


def ssa_flex(q, k, v, sel, block, num_q_per_kv):
    """Fused block-sparse attention over the selected blocks (O(n*kappa); the near-linear path).
    Requires n % block == 0 (same invariant as subq_route)."""
    B, NQ, n, d = q.shape
    nb = sel.shape[-1]
    assert n % block == 0
    if torch.cuda.is_available():
        # torch's compiled flex kernel requires the mask block size to be a multiple of its
        # BLOCK_M/BLOCK_N tiles (128/64 for fp16); a 32-block mask fails at compile time with
        # "Q and KV block size must be divisible by BLOCK_M and BLOCK_N". Fail fast here instead.
        assert block % 128 == 0, f"flex kernel needs routing block a multiple of 128, got block={block}"
    sel_q = sel.repeat_interleave(num_q_per_kv, dim=1)              # (B,NQ,nb,nb)
    kv_num = sel_q.sum(-1).to(torch.int32)
    kv_idx = torch.argsort(sel_q.int(), dim=-1, descending=True, stable=True).to(torch.int32)

    def mm(b, h, qpos, kpos):
        return kpos <= qpos                                       # token-level causal inside blocks

    bm = BlockMask.from_kv_blocks(kv_num, kv_idx, BLOCK_SIZE=block, mask_mod=mm)
    kv = k.repeat_interleave(num_q_per_kv, dim=1)
    vv = v.repeat_interleave(num_q_per_kv, dim=1)
    return _flex_kernel()(q, kv, vv, block_mask=bm, scale=d ** -0.5)


def routing_stats(sel, r, rout):
    """Research diagnostics: sparsity (selected fraction of routable), routing entropy, load balance."""
    nvis = rout.sum(-1) + 1                              # routable blocks + own block (always kept)
    sparsity = (sel.sum(-1).float() / nvis).mean()
    rv = r.masked_fill(~rout[None, None], float('-inf'))
    p = torch.softmax(rv, dim=-1, dtype=torch.float32)
    ent = -(p * torch.log(p.clamp(min=1e-9)))
    ent = torch.nan_to_num(ent, nan=0.0)                     # rows with no routable block contribute 0
    entropy = ent.sum(-1).mean()
    frac = sel.float().mean(dim=2)                                # (B,NKV,nb) share of query-blocks per key-block
    load = (sel.shape[-1] * (frac ** 2).sum(-1)).mean()           # inverse participation ratio; ~1 uniform
    return sparsity, entropy, load


class SubQAttention(nn.Module):
    def __init__(self, config: nanoSubQConfig):
        super().__init__()
        self.num_q_heads = config.num_q_heads
        self.num_kv_heads = config.num_kv_heads
        self.num_q_per_kv = config.num_q_heads // config.num_kv_heads
        self.head_dim = config.d_model // config.num_q_heads
        self.block = config.block
        self.top_c = config.top_c
        self.local = config.local
        self.beta = config.beta
        self.attn_impl = config.attn_impl if _FLEX_OK else 'masked'
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q_rot = apply_rotary_emb(q, cos[:, :, :S, :], sin[:, :, :S, :])
        k_rot = apply_rotary_emb(k, cos[:, :, :S, :], sin[:, :, :S, :])
        sel, r, rout = subq_route(q_rot, k_rot, self.block, self.top_c, self.local, self.beta)
        if self.attn_impl == 'flex':
            out = ssa_flex(q_rot, k_rot, v, sel, self.block, self.num_q_per_kv)
        else:
            out = ssa_masked(q_rot, k_rot, v, sel, self.block, self.num_q_per_kv)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        sparsity, entropy, load = routing_stats(sel, r, rout)
        return self.out_proj(out), sparsity, entropy, load


class Block(nn.Module):
    def __init__(self, config: nanoSubQConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = SubQAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model, bias=False),
        )

    def forward(self, x, cos, sin):
        attn_out, sparsity, entropy, load = self.attn(self.ln1(x), cos, sin)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, sparsity, entropy, load


class nanoSubQ(nn.Module):
    def __init__(self, config: nanoSubQConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        head_dim = config.d_model // config.num_q_heads
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(config.max_seq_len).float()
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos', emb.cos().unsqueeze(0).unsqueeze(0), persistent=True)
        self.register_buffer('sin', emb.sin().unsqueeze(0).unsqueeze(0), persistent=True)

    def forward(self, idx, targets=None):
        B, S = idx.shape
        x = self.tok_emb(idx)
        cos = self.cos[:, :, :S, :]
        sin = self.sin[:, :, :S, :]
        tot = [0.0, 0.0, 0.0]
        for layer in self.layers:
            x, sparsity, entropy, load = layer(x, cos, sin)
            tot[0] = tot[0] + sparsity
            tot[1] = tot[1] + entropy
            tot[2] = tot[2] + load
        x = self.ln_f(x)
        logits = self.lm_head(x)
        sparsity = (tot[0] / self.config.num_layers).unsqueeze(0)
        entropy = (tot[1] / self.config.num_layers).unsqueeze(0)
        load = (tot[2] / self.config.num_layers).unsqueeze(0)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100,
                label_smoothing=self.config.label_smoothing,
            ).unsqueeze(0)
        return logits, loss, sparsity, entropy, load
