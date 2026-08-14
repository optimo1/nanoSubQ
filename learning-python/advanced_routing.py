import torch
import math
import torch.nn.functional as F
import torch.nn as nn

def generate_unified_mask(
    seq_len: int,
    window_size: int,
    gate_mask: torch.Tensor,
    penalty_value: float = -1000.0,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    row_idx = torch.arange(seq_len, device=device).unsqueeze(1)
    col_idx = torch.arange(seq_len, device=device).unsqueeze(0)
    distance = row_idx - col_idx

    causal_mask = distance < 0
    sliding_window_mask = (distance >= 0) & (distance < window_size)

    router_keep = gate_mask.view(1, seq_len)
    router_penalty = (1.0 - router_keep) * penalty_value

    window_override = (~sliding_window_mask).float()
    effective_penalty = router_penalty * window_override

    return effective_penalty.masked_fill(causal_mask, float('-inf'))


class TemperatureScheduler:
    """Computes exponential temperature decay T -> t_min across training steps."""
    def __init__(
        self,
        model: nn.Module,
        t_max: float = 2.0,
        t_min: float = 0.1,
        total_steps: int = 1000
    ):
        self.model = model
        self.t_max = t_max
        self.t_min = t_min
        self.total_steps = total_steps

        # Formula: t_min = t_max * exp(-decay_rate * total_steps)
        self.decay_rate = -math.log(t_min / t_max) / total_steps

    def step(self, current_step: int) -> float:
        if current_step >= self.total_steps:
            new_t = self.t_min
        else:
            new_t = self.t_max * math.exp(-self.decay_rate * current_step)

        # Update all Router instances found within the model
        for module in self.model.modules():
            if isinstance(module, Router):
                module.set_temperature(new_t)

        return new_t


class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, raw_scores: torch.Tensor, hard_mask: torch.Tensor) -> torch.Tensor:
        return hard_mask

    @staticmethod
    def backward(ctx, *grad_output: torch.Tensor):
        return grad_output[0], None


class Router(nn.Module):
    def __init__(
        self, 
        n_embd: int, 
        rank: int = 4, 
        base_k: int = 4, 
        block_size: int = 4,
        init_temperature: float = 2.0,
        min_temperature: float = 0.1, 
        alpha: float = 0.01
    ):
        super().__init__()
        self.base_k = base_k
        self.block_size = block_size
        self.gate_down = nn.Linear(n_embd, rank, bias=False)
        self.gate_up = nn.Linear(rank, 1, bias=False)
        self.temperature = init_temperature
        self.min_temperature = min_temperature
        self.alpha = alpha

    def set_temperature(self, new_temp: float):
        """Safely updates active temperature without dropping below min_temperature."""
        self.temperature = max(new_temp, self.min_temperature)

    def calculate_k(self, seq_len: int) -> int:
        if seq_len <= self.block_size:
            return seq_len

        log_factor = math.log2(seq_len)
        raw_k = int(self.base_k * log_factor)
        optimized_k = ((raw_k + self.block_size - 1) // self.block_size) * self.block_size
        return min(optimized_k, seq_len)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        S, W, C = x.size()

        bottleneck = self.gate_down(x)
        raw_scores = self.gate_up(bottleneck)

        # Temperature scales raw scores before Sigmoid
        # T = 2.0 -> scores pull toward 0.5 (exploration, high gradient)
        # T -> 0.1 -> scores push toward 0.0 or 1.0 (deterministic decisions)
        norm_scores = torch.sigmoid(raw_scores / self.temperature)
        P = norm_scores.squeeze(-1).mean(dim=0)
        
        k = self.calculate_k(W)

        topk_values, _ = torch.topk(norm_scores, k=k, dim=1, largest=True, sorted=False)
        threshold = topk_values[:, -1:, :]

        hard_mask = (norm_scores >= threshold).float()
        f = hard_mask.squeeze(-1).mean(dim=0)
        
        final_gate = STE.apply(norm_scores, hard_mask)
        aux_loss = self.alpha * W * torch.sum(f * P)

        assert isinstance(final_gate, torch.Tensor)
        return final_gate, aux_loss


def apply_rotary_emb(x, cos, sin):
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
        self.router = Router(n_embd=d_model, base_k=4, block_size=4, rank=4, alpha=0.01)

        self.q_proj = nn.Linear(d_model, num_q_heads * self.head_dim)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model)

        self.layer_block = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, cos, sin) -> tuple[torch.Tensor, torch.Tensor]:
        shortcut = x
        seq_len = x.size(0)

        x_batched = x.unsqueeze(0)
        gate_mask, aux_loss = self.router(x_batched)
        gate_mask = gate_mask.squeeze(0)

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.view(seq_len, self.num_q_heads, self.head_dim).transpose(0, 1)
        K = K.view(seq_len, self.num_kv_heads, self.head_dim).transpose(0, 1)
        V = V.view(seq_len, self.num_kv_heads, self.head_dim).transpose(0, 1)

        Q = apply_rotary_emb(Q, cos, sin)
        K = apply_rotary_emb(K, cos, sin)

        K = K.repeat_interleave(self.num_q_per_kv, dim=0)
        V = V.repeat_interleave(self.num_q_per_kv, dim=0)

        raw_scores = torch.matmul(Q, K.transpose(-2, -1))
        scaled_scores = raw_scores / (self.head_dim ** 0.5)

        unified_mask = generate_unified_mask(
            seq_len=seq_len,
            window_size=self.window_size,
            gate_mask=gate_mask,
            penalty_value=-1000.0,
            device=x.device
        )

        masked_scores = scaled_scores + unified_mask

        attention_weights = F.softmax(masked_scores, dim=-1)
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
        
        attention_output = torch.matmul(attention_weights, V)

        stitched_output = attention_output.transpose(0, 1).contiguous().view(seq_len, self.d_model)
        blended_output = self.out_proj(stitched_output)

        math_output = self.layer_block(blended_output)
        combined_output = math_output + shortcut

        return self.norm(combined_output), aux_loss


class Main(nn.Module):
    def __init__(self, d_model, num_layers, num_q_heads, num_kv_heads, window_size: int = 8, max_seq_len=2048):
        super().__init__()
        self.layer = nn.ModuleList([
            Tool(d_model, num_q_heads, num_kv_heads, window_size=window_size) for _ in range(num_layers)
        ])
        
        self.head_dim = d_model // num_q_heads
        self.precompute_rope(max_seq_len, self.head_dim)

    def precompute_rope(self, max_seq_len, head_dim):
        positions = torch.arange(max_seq_len).unsqueeze(1)
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        angles = positions * inv_freq
        
        self.register_buffer("cos_cached", torch.cos(angles).unsqueeze(0), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles).unsqueeze(0), persistent=False)

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        total_aux_loss = torch.tensor(0.0, device=x.device)
        for layer in self.layer:
            x, aux_loss = layer(x, self.cos_cached, self.sin_cached)
            total_aux_loss = total_aux_loss + aux_loss
        return x, total_aux_loss


def verify_causal_leakage(model: nn.Module, seq_len: int = 16, d_model: int = 16):
    model.zero_grad()
    
    x = torch.randn(seq_len, d_model, requires_grad=True)
    output, aux_loss = model(x)
    
    target_idx = 5
    grad_projection = torch.randn_like(output[target_idx])
    target_loss = (output[target_idx] * grad_projection).sum()
    target_loss.backward()
    
    input_grads = x.grad
    past_grads = input_grads[:target_idx + 1].norm().item()
    future_grads = input_grads[target_idx + 1:].norm().item()
    
    print(f"\n--- Causal Leakage Test (Target Token: {target_idx}) ---")
    print(f"Past & Current Input Grad Norm (Indices 0..{target_idx}): {past_grads:.6f}")
    print(f"Future Input Grad Norm (Indices {target_idx + 1}..{seq_len - 1}): {future_grads:.6f}")
    
    assert past_grads > 0.0, "❌ ERROR: Past gradient is zero! Check tensor connection."
    assert future_grads == 0.0, "❌ CRITICAL: Causal leakage detected!"
    print("✅ SUCCESS: Zero causal leakage verified! Future gradients are strictly 0.0.")


if __name__ == "__main__":
    torch.manual_seed(686)
    
    model = Main(d_model=16, num_layers=4, num_q_heads=8, num_kv_heads=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    
    TOTAL_STEPS = 50
    scheduler = TemperatureScheduler(
        model, 
        t_max=2.0, 
        t_min=0.1, 
        total_steps=TOTAL_STEPS
    )

    print("=== Training Simulation with Temperature Annealing ===")
    for step in range(TOTAL_STEPS + 1):
        x = torch.randn(16, 16, requires_grad=True)
        
        # 1. Decay temperature across all Routers
        current_temp = scheduler.step(step)
        
        # 2. Forward pass
        output, total_aux_loss = model(x)
        loss = output.sum() + total_aux_loss
        
        # 3. Backward pass & step
        optimizer.zero_grad()
        loss.backward()
        
        down_grad = model.layer[0].router.gate_down.weight.grad.norm().item()
        up_grad = model.layer[0].router.gate_up.weight.grad.norm().item()
        
        optimizer.step()

        if step % 10 == 0:
            print(
                f"Step {step:2d} | "
                f"Temp: {current_temp:.4f} | "
                f"Total Loss: {loss.item():.4f} | "
                f"Gate Down Grad: {down_grad:.6f}"
            )

    # Verify causal integrity post-training
    verify_causal_leakage(model)