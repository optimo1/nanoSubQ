import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class nanoSubQConfig:
    vocab_size: int = 50304
    max_seq_len: int = 1024
    d_model: int = 384
    num_layers: int = 6
    num_q_heads: int = 6
    num_kv_heads: int = 2
    block: int = 32     # routing key-block size (real SubQ routes per block, not per token)
    top_c: int = 8      # top key-blocks selected per query (causal)
    local: int = 1      # always keep own block + this many preceding blocks
    dropout: float = 0.0

def apply_rotary_emb(x, cos, sin):
    d_half = x.shape[-1] // 2
    x1 = x[..., :d_half]
    x2 = x[..., d_half:]
    rotated_x = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rotated_x * sin)

def ssa_masked_gqa(q, k, v, block, top_c, local, num_q_per_kv):
    """Real-SubQ block-cumulant routing, decided per KV head (GQA), applied as an additive
    mask before exact softmax. No learnable router, no multiply/renormalize (the part that
    made training diverge). q: (B, num_q_heads, n, d); k, v: (B, num_kv_heads, n, d).
    Returns (out (B, num_q_heads, n, d), sel_frac)."""
    B, NQ, n, d = q.shape
    NKV = k.shape[1]
    nb = (n + block - 1) // block
    pad = nb * block - n
    kk = F.pad(k, (0, 0, 0, pad)) if pad else k

    # per-KV-head routing statistics, computed in fp32 (fp16-safe on T4)
    kb = kk.float().view(B, NKV, nb, block, d)
    mu = kb.mean(3)                                          # block mean       (2nd-cumulant part 1)
    var = kb.var(3, unbiased=False)                          # block spread     (2nd-cumulant part 2)
    qg = q.float().view(B, NKV, num_q_per_kv, n, d).mean(2)  # one query per KV head (group mean)
    r = torch.einsum('bhnd,bhkd->bhnk', qg, mu) + 0.5 * torch.einsum('bhnd,bhkd->bhnk', qg * qg, var)

    # a block is routable only if it is fully before this query position
    qpos = torch.arange(n, device=q.device)
    kblk = torch.arange(nb, device=q.device)
    causal_blk = qpos[:, None] >= kblk[None, :] * block
    r = r.masked_fill(~causal_blk[None, None], float('-inf'))

    # top-c routable blocks per query, per KV head
    sel = torch.zeros(B, NKV, n, nb, dtype=torch.bool, device=q.device)
    sel.scatter_(-1, r.topk(min(top_c, nb), dim=-1).indices, True)
    # always keep the local window: own block + `local` preceding blocks
    diff = (qpos // block)[:, None] - kblk[None, :]
    sel = sel | ((diff >= 0) & (diff <= local))[None, None]

    # GQA: expand per-KV-head selection to the Q heads it serves
    sel = sel.repeat_interleave(num_q_per_kv, dim=1)         # (B, NQ, n, nb)

    keyblk = torch.arange(n, device=q.device) // block
    keymask = sel.gather(-1, keyblk.view(1, 1, 1, n).expand(B, NQ, n, n))
    mask = keymask & (qpos[None, :] <= qpos[:, None])[None, None]   # + token-level causal

    kv = k.repeat_interleave(num_q_per_kv, dim=1)
    vv = v.repeat_interleave(num_q_per_kv, dim=1)
    S = (q @ kv.transpose(-1, -2)) * (d ** -0.5)
    w = torch.softmax(S.masked_fill(~mask, float('-inf')), dim=-1, dtype=torch.float32).to(v.dtype)
    out = torch.matmul(w, vv)
    return out, sel.float().mean()

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
        out, sel_frac = ssa_masked_gqa(q_rot, k_rot, v, self.block, self.top_c, self.local, self.num_q_per_kv)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out), sel_frac

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
        attn_out, sel_frac = self.attn(self.ln1(x), cos, sin)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, sel_frac

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
        total_sparsity = 0.0
        for layer in self.layers:
            x, sel_frac = layer(x, cos, sin)
            total_sparsity = total_sparsity + sel_frac
        x = self.ln_f(x)
        logits = self.lm_head(x)
        avg_sparsity = (total_sparsity / self.config.num_layers).unsqueeze(0)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            ).unsqueeze(0)
        return logits, loss, avg_sparsity
