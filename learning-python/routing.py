import torch
import math
import torch.nn.functional as F
import torch.nn as nn

class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, raw_scores: torch.Tensor, hard_mask: torch.Tensor) -> torch.Tensor:
        return hard_mask

    @staticmethod
    def backward(ctx, *grad_output: torch.Tensor):
        return grad_output[0], None


class Router(nn.Module):
    def __init__(self, n_embd: int, rank: int = 4, base_k: int = 4, block_size: int = 4, temperature: float = 1.0):
        super().__init__()
        self.base_k = base_k
        self.block_size = block_size
        self.gate_down = nn.Linear(n_embd, rank, bias=False)
        self.gate_up = nn.Linear(rank, 1, bias=False)
        self.temperature = temperature

    def calculate_k(self, seq_len: int) -> int:
        if seq_len <= self.block_size:
            return seq_len

        log_factor = math.log2(seq_len)
        raw_k = int(self.base_k * log_factor)
        optimized_k = ((raw_k + self.block_size - 1) // self.block_size) * self.block_size
        return min(optimized_k, seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        S, W, C = x.size()

        bottleneck = self.gate_down(x)
        raw_scores = self.gate_up(bottleneck)

        norm_scores = F.softmax(raw_scores / self.temperature, dim=1)
        
        k = self.calculate_k(W)

        topk_values, _ = torch.topk(norm_scores, k=k, dim=1, largest=True, sorted=False)
        threshold = topk_values[:, -1:, :]

        # Clean binary mask: 1.0 for keep, 0.0 for drop.
        hard_mask = (norm_scores >= threshold).float()
        
        # STE cleanly passes gradients back to raw_scores
        final_gate = STE.apply(norm_scores, hard_mask)

        assert isinstance(final_gate, torch.Tensor)
        return final_gate


def apply_rotary_emb(x, cos, sin):
    # Slice dynamically to match the current sequence length
    seq_len = x.shape[1]
    cos = cos[:, :seq_len, :]
    sin = sin[:, :seq_len, :]

    head_dim = x.shape[-1]
    x_left = x[..., :head_dim // 2]
    x_right = x[..., head_dim // 2:]

    rotates_left = x_left * cos - x_right * sin
    rotates_right = x_right * cos + x_left * sin

    return torch.cat([rotates_left, rotates_right], dim=-1)


class Tool(nn.Module):
    def __init__(self, d_model, num_q_heads, num_kv_heads, window_size: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_q_heads
        self.window_size = window_size
        
        assert d_model % num_q_heads == 0, "Should divide evenly"
        assert num_q_heads % num_kv_heads == 0, "Should divide evenly"

        self.num_q_per_kv = num_q_heads // num_kv_heads
        self.router = Router(n_embd=d_model, base_k=4, block_size=4, rank = 4)

        self.q_proj = nn.Linear(d_model, num_q_heads * self.head_dim)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model)

        self.layer_block = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, cos, sin):
        shortcut = x
        seq_len = x.size(0)

        # 1. Get mask from router: (seq_len, 1)
        x_batched = x.unsqueeze(0)
        gate_mask = self.router(x_batched).squeeze(0)

        # 2. Project Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.view(seq_len, self.num_q_heads, self.head_dim).transpose(0, 1)
        K = K.view(seq_len, self.num_kv_heads, self.head_dim).transpose(0, 1)
        V = V.view(seq_len, self.num_kv_heads, self.head_dim).transpose(0, 1)

        # 3. Apply Cached RoPE
        Q = apply_rotary_emb(Q, cos, sin)
        K = apply_rotary_emb(K, cos, sin)

        K = K.repeat_interleave(self.num_q_per_kv, dim=0)
        V = V.repeat_interleave(self.num_q_per_kv, dim=0)

        raw_scores = torch.matmul(Q, K.transpose(-2, -1))
        scaled_scores = raw_scores / (self.head_dim ** 0.5)

        # Causal mask + Router mask
        row_idx = torch.arange(seq_len, device = x.device).unsqueeze(1)
        col_idx = torch.arange(seq_len, device = x.device).unsqueeze(0)

        causal_mask = col_idx > row_idx
        sliding_window_mask = (row_idx - col_idx >=0)&(row_idx - col_idx < self.window_size)

        masked_scores = scaled_scores.masked_fill(causal_mask, float('-inf'))

        routing_penalty = (1.0 - gate_mask.transpose(0, 1)) * -1e4

        window_override = sliding_window_mask.float()
        effective_penalty = routing_penalty * (1.0 - window_override)

        masked_scores = masked_scores + effective_penalty

        attention_weights = F.softmax(masked_scores, dim=-1)
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
        
        attention_output = torch.matmul(attention_weights, V)

        stitched_output = attention_output.transpose(0, 1).contiguous().view(seq_len, self.d_model)
        blended_output = self.out_proj(stitched_output)

        math_output = self.layer_block(blended_output)
        combined_output = math_output + shortcut

        return self.norm(combined_output)


class Main(nn.Module):
    def __init__(self, d_model, num_layers, num_q_heads, num_kv_heads, window_size: int = 8, max_seq_len=2048):
        super().__init__()
        self.layer = nn.ModuleList([
            Tool(d_model, num_q_heads, num_kv_heads, window_size=window_size) for _ in range(num_layers)
        ])
        
        # Precompute RoPE frequencies once in __init__
        self.head_dim = d_model // num_q_heads
        self.precompute_rope(max_seq_len, self.head_dim)

    def precompute_rope(self, max_seq_len, head_dim):
        positions = torch.arange(max_seq_len).unsqueeze(1)
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        angles = positions * inv_freq
        
        # Register as buffers so they move to the correct device automatically
        self.register_buffer("cos_cached", torch.cos(angles).unsqueeze(0), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles).unsqueeze(0), persistent=False)

    def forward(self, x):
        # Pass the precomputed buffers to the layers
        for layer in self.layer:
            x = layer(x, self.cos_cached, self.sin_cached)
        return x


if __name__ == "__main__":
    torch.manual_seed(664)
    x_new = torch.randn(16, 16, requires_grad=True)
    Kaka_new = Main(16, 4, 8, 2)
        
    output_new = Kaka_new(x_new)
    loss_new = output_new.sum()
    loss_new.backward()
        
    down_grad = Kaka_new.layer[0].router.gate_down.weight.grad
    up_grad = Kaka_new.layer[0].router.gate_up.weight.grad

    print("Down Projection Grad Norm:", down_grad.norm().item())
    print("Up Projection Grad Norm:", up_grad.norm().item())