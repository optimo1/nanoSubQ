import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureScheduler:
    """Anneals router temperature exponentially from t_max to t_min across training steps."""

    def __init__(
        self,
        model: nn.Module,
        t_max: float = 2.0,
        t_min: float = 0.1,
        total_steps: int = 1000,
    ):
        self.model = model
        self.t_max = t_max
        self.t_min = t_min
        self.total_steps = total_steps
        self.decay_rate = -math.log(t_min / t_max) / total_steps

    def step(self, current_step: int) -> float:
        if current_step >= self.total_steps:
            new_t = self.t_min
        else:
            new_t = self.t_max * math.exp(-self.decay_rate * current_step)

        for module in self.model.modules():
            if isinstance(module, PerKVHeadRouter):
                module.set_temperature(new_t)

        return new_t


class PerKVHeadRouter(nn.Module):
    """Evaluates long-range historical candidates per KV head [B, H_kv, S]."""

    def __init__(
        self,
        head_dim: int,
        num_kv_heads: int,
        rank: int = 4,
        base_k: int = 4,
        block_size: int = 4,
        init_temperature: float = 2.0,
        min_temperature: float = 0.1,
        alpha: float = 0.01,
    ):
        super().__init__()
        self.base_k = base_k
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads

        self.gate_down = nn.Linear(head_dim, rank, bias=False)
        self.gate_up = nn.Linear(rank, 1, bias=False)

        self.temperature = init_temperature
        self.min_temperature = min_temperature
        self.alpha = alpha

    def set_temperature(self, new_temp: float):
        self.temperature = max(new_temp, self.min_temperature)

    def calculate_k(self, seq_len: int) -> int:
        if seq_len <= self.block_size:
            return seq_len

        log_factor = math.log2(seq_len)
        raw_k = int(self.base_k * log_factor)
        optimized_k = (
            (raw_k + self.block_size - 1) // self.block_size
        ) * self.block_size
        return min(optimized_k, seq_len)

    def calculate_routing_entropy(
        self, norm_scores: torch.Tensor, eps: float = 1e-7
    ) -> torch.Tensor:
        """Computes stable average binary Shannon entropy (in bits) over gate scores."""
        p = torch.clamp(norm_scores, eps, 1.0 - eps)
        entropy = -p * torch.log2(p) - (1.0 - p) * torch.log2(1.0 - p)
        entropy = torch.nan_to_num(entropy, nan=0.0)
        return entropy.mean()

    def forward(
        self, k_states: torch.Tensor, window_size: int = 8
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # k_states: [B, H_kv, S, D_head]
        B, H_kv, S, D_head = k_states.shape

        bottleneck = self.gate_down(k_states)                       # [B, H_kv, S, rank]
        raw_scores = self.gate_up(bottleneck).squeeze(-1)           # [B, H_kv, S]

        norm_scores = torch.sigmoid(raw_scores / self.temperature)
        P = norm_scores.mean()

        routing_entropy = self.calculate_routing_entropy(norm_scores)
        hist_len = max(0, S - window_size)

        local_indices = (
            torch.arange(hist_len, S, device=k_states.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(B, H_kv, -1)
        )                                                           # [B, H_kv, W]

        if hist_len > 0:
            hist_scores = norm_scores[:, :, :hist_len]              # [B, H_kv, hist_len]
            k_global = self.calculate_k(hist_len)

            topk_scores, global_indices = torch.topk(
                hist_scores, k=k_global, dim=-1, largest=True, sorted=True
            )                                                       # [B, H_kv, K_global]

            local_scores = norm_scores[:, :, hist_len:]             # [B, H_kv, W]

            combined_indices = torch.cat([global_indices, local_indices], dim=-1)
            combined_scores = torch.cat([topk_scores, local_scores], dim=-1)
        else:
            combined_indices = local_indices
            combined_scores = norm_scores

        # Chronological sort per head
        selected_indices, sort_perm = torch.sort(combined_indices, dim=-1)
        selected_scores = torch.gather(combined_scores, dim=-1, index=sort_perm)

        # Aux loss
        k_total = selected_indices.size(-1)
        f = torch.tensor(k_total / S, device=k_states.device, dtype=k_states.dtype)
        aux_loss = self.alpha * (f * P)

        # Straight-Through Estimator (STE) dynamic thresholding
        hard_weights = (selected_scores > 0.5).to(dtype=selected_scores.dtype)
        selected_weights = selected_scores + (hard_weights - selected_scores).detach()

        return selected_indices, selected_weights, aux_loss, routing_entropy


def apply_rotary_emb_indexed(
    x: torch.Tensor,
    cos_cached: torch.Tensor,
    sin_cached: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """
    Applies RoPE per head using direct tensor indexing.
    x: [B, H, K, D_head]
    indices: [B, H, K]
    """
    B, H, K, D_head = x.shape
    cos_table = cos_cached.squeeze(0).squeeze(0)  # [Max_S, D_head/2]
    sin_table = sin_cached.squeeze(0).squeeze(0)  # [Max_S, D_head/2]

    # Direct multi-dimensional indexing
    cos = cos_table[indices]                      # [B, H, K, D_head/2]
    sin = sin_table[indices]                      # [B, H, K, D_head/2]

    x_left = x[..., : D_head // 2]
    x_right = x[..., D_head // 2 :]

    rotates_left = x_left * cos - x_right * sin
    rotates_right = x_right * cos + x_left * sin

    return torch.cat([rotates_left, rotates_right], dim=-1)


class BatchedSparseAttentionBlock(nn.Module):
    """Processes dynamic sparse tokens using GQA & Per-KV Head Routing."""

    def __init__(
        self, d_model: int, num_q_heads: int, num_kv_heads: int, window_size: int = 8
    ):
        super().__init__()
        self.d_model = d_model
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_q_heads
        self.window_size = window_size

        assert d_model % num_q_heads == 0, "d_model must be divisible by num_q_heads"
        assert (
            num_q_heads % num_kv_heads == 0
        ), "num_q_heads must be divisible by num_kv_heads"

        self.num_q_per_kv = num_q_heads // num_kv_heads

        self.router = PerKVHeadRouter(
            head_dim=self.head_dim,
            num_kv_heads=num_kv_heads,
            base_k=4,
            block_size=4,
            rank=4,
            alpha=0.01,
        )

        self.q_proj = nn.Linear(d_model, num_q_heads * self.head_dim)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model)

        self.layer_block = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, S, D = x.size()
        shortcut = x

        # 1. Full Sequence Projections
        q_full = (
            self.q_proj(x)
            .view(B, S, self.num_q_heads, self.head_dim)
            .transpose(1, 2)
        )                                                           # [B, H_q, S, D_head]
        k_full = (
            self.k_proj(x)
            .view(B, S, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )                                                           # [B, H_kv, S, D_head]
        v_full = (
            self.v_proj(x)
            .view(B, S, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )                                                           # [B, H_kv, S, D_head]

        # 2. Per-KV Head Router Selection
        kv_indices, kv_weights, aux_loss, routing_entropy = self.router(
            k_full, window_size=self.window_size
        )                                                           # [B, H_kv, K]

        K_len = kv_indices.size(-1)

        # 3. Gather KV States
        kv_gather_idx = kv_indices.unsqueeze(-1).expand(
            B, self.num_kv_heads, K_len, self.head_dim
        )
        k_sparse = torch.gather(k_full, dim=2, index=kv_gather_idx)  # [B, H_kv, K, D_head]
        v_sparse = torch.gather(v_full, dim=2, index=kv_gather_idx)  # [B, H_kv, K, D_head]

        # Expand KV Head Indices & Weights to Query Head Space
        q_indices = kv_indices.repeat_interleave(self.num_q_per_kv, dim=1)  # [B, H_q, K]
        q_weights = kv_weights.repeat_interleave(self.num_q_per_kv, dim=1)  # [B, H_q, K]

        # Gather Query States mapped to matched KV selections
        q_gather_idx = q_indices.unsqueeze(-1).expand(
            B, self.num_q_heads, K_len, self.head_dim
        )
        q_sparse = torch.gather(q_full, dim=2, index=q_gather_idx)   # [B, H_q, K, D_head]

        # 4. RoPE
        q = apply_rotary_emb_indexed(q_sparse, cos, sin, q_indices)
        k = apply_rotary_emb_indexed(k_sparse, cos, sin, kv_indices)

        # Expand KV to match Query Head count for GQA attention calculation
        k = k.repeat_interleave(self.num_q_per_kv, dim=1)            # [B, H_q, K, D_head]
        v = v_sparse.repeat_interleave(self.num_q_per_kv, dim=1)     # [B, H_q, K, D_head]

        # 5. Scaled Dot-Product Attention
        raw_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)

        # Strict Causal Masking on original sequence position indices
        # Mask out target positions (keys) that appear strictly after query positions
        kv_indices_q = kv_indices.repeat_interleave(self.num_q_per_kv, dim=1)
        causal_mask = q_indices.unsqueeze(-1) < kv_indices_q.unsqueeze(-2)  # [B, H_q, K, K]

        masked_scores = raw_scores.masked_fill(causal_mask, float("-inf"))
        attention_weights = F.softmax(masked_scores, dim=-1)
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)

        attention_output = torch.matmul(attention_weights, v)       # [B, H_q, K, D_head]

        # Apply STE weights
        weighted_output = attention_output * q_weights.unsqueeze(-1)

        # 6. Scatter-add back to full sequence query space [B, H_q, S, D_head]
        output_full = torch.zeros_like(q_full)
        output_full.scatter_add_(dim=2, index=q_gather_idx, src=weighted_output)

        # Output projection and feedforward
        stitched = (
            output_full.transpose(1, 2)
            .contiguous()
            .view(B, S, self.d_model)
        )
        blended = self.out_proj(stitched)
        math_output = self.layer_block(blended)

        final_output = math_output + shortcut
        return self.norm(final_output), aux_loss, routing_entropy


class BatchedSparseTransformer(nn.Module):
    """Full batched Transformer model stacking multiple BatchedSparseAttentionBlocks."""

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_q_heads: int,
        num_kv_heads: int,
        window_size: int = 8,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.layer = nn.ModuleList(
            [
                BatchedSparseAttentionBlock(
                    d_model, num_q_heads, num_kv_heads, window_size=window_size
                )
                for _ in range(num_layers)
            ]
        )

        self.head_dim = d_model // num_q_heads
        self.precompute_rope(max_seq_len, self.head_dim)

    def precompute_rope(self, max_seq_len: int, head_dim: int):
        positions = torch.arange(max_seq_len).unsqueeze(1)
        inv_freq = 1.0 / (
            10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        angles = positions * inv_freq

        self.register_buffer(
            "cos_cached", torch.cos(angles).unsqueeze(0).unsqueeze(0), persistent=False
        )                                                           # [1, 1, S, D/2]
        self.register_buffer(
            "sin_cached", torch.sin(angles).unsqueeze(0).unsqueeze(0), persistent=False
        )                                                           # [1, 1, S, D/2]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        total_aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        total_entropy = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        for layer_module in self.layer:
            x, aux_loss, routing_entropy = layer_module(x, self.cos_cached, self.sin_cached)
            total_aux_loss = total_aux_loss + aux_loss
            total_entropy = total_entropy + routing_entropy

        avg_entropy = total_entropy / len(self.layer)
        return x, total_aux_loss, avg_entropy


def verify_batched_causal_leakage(
    model: nn.Module, batch_size: int = 2, seq_len: int = 16, d_model: int = 16
):
    model.zero_grad()

    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    output, _, _ = model(x)

    target_batch_idx = 0
    target_token_idx = 5

    grad_projection = torch.randn_like(output[target_batch_idx, target_token_idx])
    target_loss = (
        output[target_batch_idx, target_token_idx] * grad_projection
    ).sum()
    target_loss.backward()

    input_grads = x.grad
    assert input_grads is not None, "Gradient extraction failed."

    target_grads = input_grads[target_batch_idx]
    past_grads = target_grads[: target_token_idx + 1].norm().item()
    future_grads = target_grads[target_token_idx + 1 :].norm().item()

    print(f"\n--- 3D Batched Causal Leakage Test ---")
    print(f"Target Batch Item: {target_batch_idx} | Target Token: {target_token_idx}")
    print(
        f"Past & Current Input Grad Norm (Indices 0..{target_token_idx}): {past_grads:.6f}"
    )
    print(
        f"Future Input Grad Norm (Indices {target_token_idx + 1}..{seq_len - 1}): {future_grads:.6f}"
    )

    assert (
        past_grads > 0.0
    ), "❌ ERROR: Past gradient is zero! Check tensor connection."
    assert future_grads == 0.0, "❌ CRITICAL: Causal leakage detected!"
    print(
        "✅ SUCCESS: Zero causal leakage verified! Future gradients are strictly 0.0."
    )


def test_routing_entropy_across_sequence_lengths(
    model: nn.Module, batch_size: int = 4, d_model: int = 16
):
    seq_lengths = [16, 32, 64, 128, 256, 512]
    model.eval()

    print("\n--- Routing Entropy Test Across Sequence Lengths ---")
    print(f"{'Seq Len':>8} | {'Selected K':>10} | {'Sparsity %':>10} | {'Avg Routing Entropy (bits)':>28}")
    print("-" * 68)

    with torch.no_grad():
        for S in seq_lengths:
            x = torch.randn(batch_size, S, d_model)
            _, _, avg_entropy = model(x)

            first_block = model.layer[0]
            k_full = (
                first_block.k_proj(x)
                .view(batch_size, S, first_block.num_kv_heads, first_block.head_dim)
                .transpose(1, 2)
            )
            selected_indices, _, _, _ = first_block.router(k_full, window_size=first_block.window_size)
            selected_k = selected_indices.size(-1)
            sparsity = (1.0 - (selected_k / S)) * 100.0

            entropy_val = avg_entropy.item()
            print(f"{S:8d} | {selected_k:10d} | {sparsity:9.1f}% | {entropy_val:28.6f}")

    print("✅ Routing Entropy Evaluation Complete.")


if __name__ == "__main__":
    torch.manual_seed(78776)

    BATCH_SIZE = 4
    SEQ_LEN = 16
    D_MODEL = 16
    TOTAL_STEPS = 50

    model = BatchedSparseTransformer(
        d_model=D_MODEL,
        num_layers=4,
        num_q_heads=8,
        num_kv_heads=2,
        window_size=8,
        max_seq_len=2048,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    scheduler = TemperatureScheduler(
        model, t_max=2.0, t_min=0.1, total_steps=TOTAL_STEPS
    )

    print("=== Training Simulation with Batched Inputs [B, S, D] ===")
    for step in range(TOTAL_STEPS + 1):
        x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL, requires_grad=True)

        current_temp = scheduler.step(step)

        output, total_aux_loss, avg_entropy = model(x)
        target = torch.randn_like(output)
        task_loss = F.mse_loss(output, target)
        loss = task_loss + total_aux_loss

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        first_layer = model.layer[0]
        assert isinstance(first_layer, BatchedSparseAttentionBlock)
        gate_down_grad = first_layer.router.gate_down.weight.grad
        assert gate_down_grad is not None

        optimizer.step()

        if step % 10 == 0:
            print(
                f"Step {step:2d} | "
                f"Temp: {current_temp:.4f} | "
                f"Total Loss: {loss.item():.4f} | "
                f"Routing Entropy: {avg_entropy.item():.4f} | "
                f"Gate Down Grad: {gate_down_grad.norm().item():.6f}"
            )

    verify_batched_causal_leakage(
        model, batch_size=BATCH_SIZE, seq_len=SEQ_LEN, d_model=D_MODEL
    )

    test_routing_entropy_across_sequence_lengths(
        model, batch_size=BATCH_SIZE, d_model=D_MODEL
    )