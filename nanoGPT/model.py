import math
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
    window_size: int = 8
    k_ratio: float = 0.5
    dropout: float = 0.0

class StraightThroughEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return (x >= 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class TemperatureScheduler:
    def __init__(self, model, t_max=2.0, t_min=0.1, total_steps=50000, **kwargs):
        self.model = model
        self.t_max = t_max
        self.t_min = t_min
        self.total_steps = total_steps

    def step(self, current_step):
        progress = min(1.0, max(0.0, current_step / self.total_steps))
        temp = self.t_max - progress * (self.t_max - self.t_min)
        
        target_model = self.model.module if hasattr(self.model, 'module') else self.model
        for block in target_model.layers:
            block.attn.temperature = temp
        return temp

def apply_rotary_emb(x, cos, sin):
    d_half = x.shape[-1] // 2
    x1 = x[..., :d_half]
    x2 = x[..., d_half:]
    rotated_x = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rotated_x * sin)

class SubQAttention(nn.Module):
    def __init__(self, config: nanoSubQConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.num_q_heads = config.num_q_heads
        self.num_kv_heads = config.num_kv_heads
        self.num_q_per_kv = config.num_q_heads // config.num_kv_heads
        self.head_dim = config.d_model // config.num_q_heads
        self.window_size = config.window_size
        self.k_ratio = config.k_ratio
        self.temperature = 2.0

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.router = nn.Linear(config.d_model, config.num_kv_heads)

    def forward(self, x, cos, sin):
        B, S, D = x.shape

        q = self.q_proj(x).view(B, S, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q_rot = apply_rotary_emb(q, cos[:, :, :S, :], sin[:, :, :S, :])
        k_rot = apply_rotary_emb(k, cos[:, :, :S, :], sin[:, :, :S, :])

        router_logits = torch.clamp(self.router(x), -10.0, 10.0) / self.temperature
        router_scores = torch.sigmoid(router_logits).transpose(1, 2)
        ste_weights = StraightThroughEstimator.apply(router_scores)

        p_fp32 = router_scores.float()
        p_clamped = torch.clamp(p_fp32, 1e-6, 1.0 - 1e-6)
        entropy = - (p_clamped * torch.log(p_clamped) + (1.0 - p_clamped) * torch.log(1.0 - p_clamped)).mean().unsqueeze(0)

        hist_len = max(0, S - self.window_size)
        
        if hist_len > 0:
            hist_scores = router_scores.detach()[:, :, :hist_len]
            k_hist = max(1, int(hist_len * self.k_ratio))
            _, topk_indices = torch.topk(hist_scores, k=k_hist, dim=-1)

            local_indices = torch.arange(hist_len, S, device=x.device).unsqueeze(0).unsqueeze(0).expand(B, self.num_kv_heads, -1)
            kv_indices = torch.cat([topk_indices, local_indices], dim=-1)
        else:
            kv_indices = torch.arange(S, device=x.device).unsqueeze(0).unsqueeze(0).expand(B, self.num_kv_heads, -1)

        kv_indices_sorted, _ = torch.sort(kv_indices, dim=-1)

        gather_idx = kv_indices_sorted.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        k_rot_sparse = torch.gather(k_rot, dim=2, index=gather_idx)
        v_sparse = torch.gather(v, dim=2, index=gather_idx)

        q_ste_weights = ste_weights.repeat_interleave(self.num_q_per_kv, dim=1)
        k_rot_sparse = k_rot_sparse.repeat_interleave(self.num_q_per_kv, dim=1)
        v_sparse = v_sparse.repeat_interleave(self.num_q_per_kv, dim=1)

        att_scores = torch.matmul(q_rot, k_rot_sparse.transpose(-2, -1)) / math.sqrt(self.head_dim)

        q_idx = torch.arange(S, device=x.device).view(1, 1, S, 1)
        kv_idx_expanded = kv_indices_sorted.repeat_interleave(self.num_q_per_kv, dim=1).unsqueeze(2)
        causal_mask = kv_idx_expanded > q_idx

        att_scores = att_scores.masked_fill(causal_mask, float('-inf'))
        att_weights = F.softmax(att_scores, dim=-1)
        att_weights = torch.nan_to_num(att_weights, nan=0.0)

        out = torch.matmul(att_weights, v_sparse)
        
        # Detach STE mask during multiplication to prevent router gradient corruption of attention states
        out = out * q_ste_weights.detach().unsqueeze(-1)
        
        out = out.transpose(1, 2).contiguous().view(B, S, D)

        return self.out_proj(out), entropy

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
        attn_out, entropy = self.attn(self.ln1(x), cos, sin)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, entropy

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
        self.register_buffer('cos', emb.cos().unsqueeze(0).unsqueeze(0), persistent=False)
        self.register_buffer('sin', emb.sin().unsqueeze(0).unsqueeze(0), persistent=False)

    def forward(self, idx, targets=None):
        B, S = idx.shape
        x = self.tok_emb(idx)

        cos = self.cos[:, :, :S, :]
        sin = self.sin[:, :, :S, :]

        total_entropy = 0.0
        for layer in self.layers:
            x, entropy = layer(x, cos, sin)
            total_entropy = total_entropy + entropy.mean()

        x = self.ln_f(x)
        logits = self.lm_head(x)

        avg_entropy = (total_entropy / self.config.num_layers).unsqueeze(0)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)).unsqueeze(0)

        return logits, loss, avg_entropy