import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

@dataclass
class nanoSubQConfig:
    num_layers: int = 12
    num_q_heads: int = 12
    num_kv_heads: int = 4
    d_model: int = 768
    max_seq_len: int = 1024
    vocab_size: int = 50304  # Padded for optimal CUDA alignment
    dropout: float = 0.0
    bias: bool = False       # Set to False to disable router and linear biases for STE stability
    clamp_val: float = 4.0   # Router logit clamp threshold

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class STERouter(nn.Module):
    """
    Straight-Through Estimator (STE) Router with Logit Clamping and Temperature Scaling.
    """
    def __init__(self, config: nanoSubQConfig):
        super().__init__()
        self.clamp_val = config.clamp_val
        self.proj = nn.Linear(config.d_model, config.num_q_heads, bias=config.bias)
        self.register_buffer("temperature", torch.tensor(2.0))

    def set_temperature(self, temp: float):
        self.temperature.fill_(temp)

    def forward(self, x):
        # 1. Linear Projection
        logits = self.proj(x)
        
        # 2. Logit Clamping to avoid Sigmoid/Softmax gradient vanishing
        logits = torch.clamp(logits, -self.clamp_val, self.clamp_val)
        
        # 3. Soft Routing Probabilities for Loss / Entropy tracking
        soft_probs = F.softmax(logits / self.temperature, dim=-1)
        
        # 4. Straight-Through Estimator discretization (Hard in forward, Soft in backward)
        hard_mask = torch.zeros_like(soft_probs).scatter_(-1, soft_probs.argmax(dim=-1, keepdim=True), 1.0)
        ste_mask = hard_mask + (soft_probs - soft_probs.detach())
        
        # Entropy tracking: H(p) = -sum(p * log(p))
        entropy = -torch.sum(soft_probs * torch.log(soft_probs + 1e-9), dim=-1).mean()
        
        return ste_mask, entropy

class CausalSelfAttention(nn.Module):
    def __init__(self, config: nanoSubQConfig):
        super().__init__()
        assert config.d_model % config.num_q_heads == 0
        self.num_q_heads = config.num_q_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.d_model // config.num_q_heads
        self.num_queries_per_kv = config.num_q_heads // config.num_kv_heads

        self.router = STERouter(config)

        self.q_proj = nn.Linear(config.d_model, config.num_q_heads * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, config.num_kv_heads * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, config.num_kv_heads * self.head_dim, bias=config.bias)
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(self, x):
        B, T, C = x.size()

        # Route sub-queries
        ste_mask, entropy = self.router(x)

        q = self.q_proj(x).view(B, T, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.num_queries_per_kv > 1:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            bias = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
            att = att.masked_fill(bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        # Apply STE mask weights to sub-queries
        y = y * ste_mask.transpose(1, 2).unsqueeze(-1)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        
        return y, entropy

class MLP(nn.Module):
    def __init__(self, config: nanoSubQConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, 4 * config.d_model, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.d_model, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x, torch.tensor(0.0, device=x.device)

class Block(nn.Module):
    def __init__(self, config: nanoSubQConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.d_model, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.d_model, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        attn_out, entropy = self.attn(self.ln_1(x))
        x = x + attn_out
        mlp_out, _ = self.mlp(self.ln_2(x))
        x = x + mlp_out
        return x, entropy

class nanoSubQ(nn.Module):
    def __init__(self, config: nanoSubQConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.d_model),
            wpe = nn.Embedding(config.max_seq_len, config.d_model),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.num_layers)]),
            ln_f = LayerNorm(config.d_model, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.num_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def set_temperature(self, temp: float):
        """ Broadcaster for decaying temperature across all STE Routers """
        for block in self.transformer.h:
            block.attn.router.set_temperature(temp)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.max_seq_len, f"Sequence length {t} exceeds max length {self.config.max_seq_len}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        total_entropy = torch.tensor(0.0, device=device)
        for block in self.transformer.h:
            x, entropy = block(x)
            total_entropy = total_entropy + entropy

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            task_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1), 
                ignore_index=-100
            )
            loss = task_loss
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss, total_entropy / len(self.transformer.h)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = []
        nodecay_params = []

        for n, p in param_dict.items():
            # Exclude 1D tensors (biases/norms) and router parameters (temperature/clamps) from weight decay
            if p.dim() < 2 or 'router' in n or 'bias' in n:
                nodecay_params.append(p)
            else:
                decay_params.append(p)

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        return optimizer