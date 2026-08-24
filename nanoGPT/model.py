import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class StraightThroughEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, threshold=0.5):
        return (scores >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class TemperatureScheduler:
    def __init__(self, model, t_max=2.0, t_min=0.1, total_steps=24000, decay_rate=3.0):
        self.model = model
        self.t_max = t_max
        self.t_min = t_min
        self.total_steps = total_steps
        self.decay_rate = decay_rate

    def step(self, current_step):
        progress = min(1.0, current_step / self.total_steps)
        temp = self.t_min + (self.t_max - self.t_min) * math.exp(-self.decay_rate * progress)
        for module in self.model.modules():
            if hasattr(module, 'temperature'):
                module.temperature = temp
        return temp

class nanoSubQConfig:
    def __init__(
        self,
        vocab_size=50304,
        max_seq_len=1024,
        d_model=384,
        num_layers=6,
        num_q_heads=6,
        num_kv_heads=2,
        window_size=8,
        k_ratio=0.4,
        dropout=0.0,
    ):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.num_q_per_kv = num_q_heads // num_kv_heads
        self.head_dim = d_model // num_q_heads
        self.window_size = window_size
        self.k_ratio = k_ratio
        self.dropout = dropout

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=1024):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x, indices):
        if indices.ndim == 2:
            indices = indices.unsqueeze(1).expand(-1, x.shape[1], -1)
        cos = self.cos_cached[indices]
        sin = self.sin_cached[indices]
        return (x * cos) + (self._rotate_half(x) * sin)

class SubQAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.num_q_heads = config.num_q_heads
        self.num_kv_heads = config.num_kv_heads
        self.num_q_per_kv = config.num_q_per_kv
        self.head_dim = config.head_dim
        self.window_size = config.window_size

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.router = nn.Linear(config.d_model, config.num_kv_heads)
        self.temperature = 1.0
        self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len)

    def forward(self, x, layer_past=None, use_cache=False):
        B, S, C = x.shape
        
        q = self.q_proj(x).view(B, S, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        router_logits = self.router(x) / self.temperature
        router_scores = torch.sigmoid(router_logits).transpose(1, 2)
        ste_weights = StraightThroughEstimator.apply(router_scores)

        p = torch.clamp(router_scores, 1e-6, 1.0 - 1e-6)
        entropy = - (p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p)).mean()

        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            
        present = (k, v) if use_cache else None
        total_kv_len = k.shape[2]

        if S == 1:
            curr_pos = total_kv_len - 1
            pos_tensor = torch.tensor([[curr_pos]], device=x.device).expand(B, -1)
            all_kv_positions = torch.arange(total_kv_len, device=x.device).unsqueeze(0).expand(B, -1)

            q_rot = self.rope(q, pos_tensor)
            k_rot = self.rope(k, all_kv_positions)

            k_rot = k_rot.repeat_interleave(self.num_q_per_kv, dim=1)
            v_exp = v.repeat_interleave(self.num_q_per_kv, dim=1)

            att_scores = torch.matmul(q_rot, k_rot.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            att_probs = F.softmax(att_scores, dim=-1)
            att_probs = torch.nan_to_num(att_probs, nan=0.0)
            att_out = torch.matmul(att_probs, v_exp)

            q_ste_weights = ste_weights.repeat_interleave(self.num_q_per_kv, dim=1)
            weighted_att_out = att_out * q_ste_weights.unsqueeze(-1)
            
            output = weighted_att_out.transpose(1, 2).reshape(B, S, C)
            output = self.out_proj(output)
            return output, entropy, present

        else:
            hist_len = max(0, S - self.window_size)
            if hist_len > 0:
                hist_scores = router_scores[:, :, :hist_len]
                k_hist = max(1, int(hist_len * self.config.k_ratio))
                _, topk_indices = torch.topk(hist_scores, k=k_hist, dim=-1)
                
                local_window_indices = torch.arange(hist_len, S, device=x.device)
                local_indices = local_window_indices.unsqueeze(0).unsqueeze(0).expand(B, self.num_kv_heads, -1)
                
                kv_indices = torch.cat([topk_indices, local_indices], dim=-1)
                kv_indices, _ = torch.sort(kv_indices, dim=-1)
            else:
                kv_indices = torch.arange(S, device=x.device).unsqueeze(0).unsqueeze(0).expand(B, self.num_kv_heads, -1)

            q_kv_indices = kv_indices.repeat_interleave(self.num_q_per_kv, dim=1)
            all_seq_indices = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)

            q_rot = self.rope(q, all_seq_indices)
            
            gather_idx = kv_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
            k_sparse = torch.gather(k, dim=2, index=gather_idx)
            v_sparse = torch.gather(v, dim=2, index=gather_idx)

            k_rot_sparse = self.rope(k_sparse, kv_indices)
            
            k_rot_sparse = k_rot_sparse.repeat_interleave(self.num_q_per_kv, dim=1)
            v_sparse = v_sparse.repeat_interleave(self.num_q_per_kv, dim=1)

            att_scores = torch.matmul(q_rot, k_rot_sparse.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            
            causal_mask = all_seq_indices.unsqueeze(1).unsqueeze(-1) < q_kv_indices.unsqueeze(2)
            att_scores = att_scores.masked_fill(causal_mask, float('-inf'))

            att_probs = F.softmax(att_scores, dim=-1)
            # Safe conversion of rows where all keys were masked out by causality
            att_probs = torch.nan_to_num(att_probs, nan=0.0)
            
            att_probs = F.dropout(att_probs, p=self.config.dropout, training=self.training)
            att_out = torch.matmul(att_probs, v_sparse)

            q_ste_weights = ste_weights.repeat_interleave(self.num_q_per_kv, dim=1)
            weighted_att_out = att_out * q_ste_weights.unsqueeze(-1)

            output = weighted_att_out.transpose(1, 2).reshape(B, S, C)
            output = self.out_proj(output)
            return output, entropy, present

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = SubQAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model, bias=False),
            nn.Dropout(config.dropout),
        )

    def forward(self, x, layer_past=None, use_cache=False):
        attn_out, entropy, present = self.attn(self.ln1(x), layer_past=layer_past, use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, entropy, present

class nanoSubQ(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.tok_emb.weight = self.head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, past_key_values=None, use_cache=False):
        x = self.tok_emb(idx)
        x = self.drop(x)

        total_entropy = x.new_tensor(0.0)
        presents = [] if use_cache else None

        for i, block in enumerate(self.blocks):
            layer_past = past_key_values[i] if past_key_values is not None else None
            x, entropy, present = block(x, layer_past=layer_past, use_cache=use_cache)
            total_entropy = total_entropy + entropy
            if use_cache:
                presents.append(present)

        x = self.ln_f(x)
        logits = self.head(x)
        
        avg_entropy = total_entropy / self.config.num_layers

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, avg_entropy