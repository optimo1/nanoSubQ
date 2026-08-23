import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureScheduler:
    def __init__(self, model: nn.Module, t_max: float = 2.0, t_min: float = 0.1, total_steps: int = 1000):
        self.model = model
        self.t_max = t_max
        self.t_min = t_min
        self.total_steps = total_steps
        self.decay_rate = -math.log(t_min / t_max) / total_steps

    def step(self, current_step: int) -> float:
        new_t = self.t_min if current_step >= self.total_steps else self.t_max * math.exp(-self.decay_rate * current_step)
        for module in self.model.modules():
            if isinstance(module, PerKVHeadRouter):
                module.set_temperature(new_t)
        return new_t


def calculate_routing_entropy(norm_scores: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    p = torch.clamp(norm_scores, eps, 1.0 - eps)
    entropy = -p * torch.log2(p) - (1.0 - p) * torch.log2(1.0 - p)
    return torch.nan_to_num(entropy, nan=0.0).mean()


def apply_rotary_emb_indexed(
    x: torch.Tensor, 
    cos_cached: torch.Tensor, 
    sin_cached: torch.Tensor, 
    indices: torch.Tensor
) -> torch.Tensor:
    cos_table = cos_cached.squeeze(0).squeeze(0)  # [max_seq_len, head_dim // 2]
    sin_table = sin_cached.squeeze(0).squeeze(0)  # [max_seq_len, head_dim // 2]

    cos = cos_table[indices]  # [B, H, K_len, head_dim // 2]
    sin = sin_table[indices]  # [B, H, K_len, head_dim // 2]

    D_head = x.size(-1)
    x_left = x[..., : D_head // 2]
    x_right = x[..., D_head // 2 :]

    rotates_left = x_left * cos - x_right * sin
    rotates_right = x_right * cos + x_left * sin

    return torch.cat([rotates_left, rotates_right], dim=-1)


class StraightThroughEstimator(nn.Module):
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

    def forward(self, soft_scores: torch.Tensor) -> torch.Tensor:
        hard_weights = (soft_scores > self.threshold).to(dtype=soft_scores.dtype)
        return soft_scores + (hard_weights - soft_scores).detach()


class PerKVHeadRouter(nn.Module):
    def __init__(
        self,
        head_dim: int,
        num_kv_heads: int,
        rank: int = 4,
        retention_ratio: float = 0.40,  # Retain 40% (prune 60%)
        min_k: int = 16,
        block_size: int = 4,
        init_temperature: float = 2.0,
        min_temperature: float = 0.1,
        alpha: float = 0.01,
        clamp_val: float = 4.0,
    ):
        super().__init__()
        self.retention_ratio = retention_ratio
        self.min_k = min_k
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.clamp_val = clamp_val

        self.gate_down = nn.Linear(head_dim, rank, bias=False)
        self.gate_up = nn.Linear(rank, 1, bias=False)

        nn.init.normal_(self.gate_down.weight, std=0.01)
        nn.init.normal_(self.gate_up.weight, std=0.01)

        self.temperature = init_temperature
        self.min_temperature = min_temperature
        self.alpha = alpha
        
        self.ste = StraightThroughEstimator(threshold=0.5)

    def set_temperature(self, new_temp: float):
        self.temperature = max(new_temp, self.min_temperature)

    def calculate_k(self, hist_len: int) -> int:
        if hist_len <= self.block_size:
            return hist_len

        raw_k = int(hist_len * self.retention_ratio)
        k_bounded = max(self.min_k, raw_k)
        k_bounded = min(k_bounded, hist_len)

        optimized_k = (
            (k_bounded + self.block_size - 1) // self.block_size
        ) * self.block_size

        return min(optimized_k, hist_len)

    def forward(
        self, k_states: torch.Tensor, window_size: int = 8
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, H_kv, S, D_head = k_states.shape

        bottleneck = self.gate_down(k_states)
        raw_scores = self.gate_up(bottleneck).squeeze(-1)

        scaled_logits = raw_scores / self.temperature
        clamped_logits = torch.clamp(scaled_logits, min=-self.clamp_val, max=self.clamp_val)
        norm_scores = torch.sigmoid(clamped_logits)

        P = norm_scores.mean()
        routing_entropy = calculate_routing_entropy(norm_scores)
        hist_len = max(0, S - window_size)

        local_indices = (
            torch.arange(hist_len, S, device=k_states.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(B, H_kv, -1)
        )

        if hist_len > 0:
            hist_scores = norm_scores[:, :, :hist_len]
            k_global = self.calculate_k(hist_len)

            topk_scores, global_indices = torch.topk(
                hist_scores, k=k_global, dim=-1, largest=True, sorted=True
            )

            local_scores = norm_scores[:, :, hist_len:]

            combined_indices = torch.cat([global_indices, local_indices], dim=-1)
            combined_scores = torch.cat([topk_scores, local_scores], dim=-1)
        else:
            combined_indices = local_indices
            combined_scores = norm_scores

        selected_indices, sort_perm = torch.sort(combined_indices, dim=-1)
        selected_scores = torch.gather(combined_scores, dim=-1, index=sort_perm)

        k_total = selected_indices.size(-1)
        f = torch.tensor(k_total / S, device=k_states.device, dtype=k_states.dtype)
        aux_loss = self.alpha * (f * P)

        selected_weights = self.ste(selected_scores)

        return selected_indices, selected_weights, aux_loss, routing_entropy


class SparseKVCache(nn.Module):
    """Pre-allocated static KV-cache buffer with sparse index routing support."""
    def __init__(self, batch_size: int, num_kv_heads: int, max_seq_len: int, head_dim: int, device: torch.device, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.batch_size = batch_size
        self.num_kv_heads = num_kv_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        
        # Static buffer pre-allocation
        shape = (batch_size, num_kv_heads, max_seq_len, head_dim)
        self.register_buffer("k_cache", torch.zeros(shape, dtype=dtype, device=device), persistent=False)
        self.register_buffer("v_cache", torch.zeros(shape, dtype=dtype, device=device), persistent=False)
        self.seq_pos = 0

    def update(self, k_val: torch.Tensor, v_val: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        k_val, v_val: [B, H_kv, T_new, D_head]
        Appends new K/V projections into the pre-allocated cache tensor.
        """
        bsz, num_heads, t_new, head_dim = k_val.shape
        end_pos = self.seq_pos + t_new
        
        self.k_cache[:, :, self.seq_pos:end_pos, :] = k_val
        self.v_cache[:, :, self.seq_pos:end_pos, :] = v_val
        self.seq_pos = end_pos

        k_active = self.k_cache[:, :, :self.seq_pos, :]
        v_active = self.v_cache[:, :, :self.seq_pos, :]
        return k_active, v_active

    def reset(self):
        """Clears sequence offset and resets cache buffers for next generation pass."""
        self.seq_pos = 0
        self.k_cache.zero_()
        self.v_cache.zero_()


class MLP(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.c_fc = nn.Linear(d_model, 4 * d_model)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class SubQAttention(nn.Module):
    def __init__(
        self, d_model: int, num_q_heads: int, num_kv_heads: int, window_size: int = 8, dropout: float = 0.0
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
            retention_ratio=0.40,
            min_k=16,
            block_size=4,
            rank=4,
            alpha=0.01,
            clamp_val=4.0,
        )

        self.q_proj = nn.Linear(d_model, num_q_heads * self.head_dim)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model)

        self.mlp = MLP(d_model, dropout=dropout)
        self.ln_1 = nn.LayerNorm(d_model)
        self.ln_2 = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, kv_cache: SparseKVCache = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_norm = self.ln_1(x)
        B, S, D = x_norm.size()

        q_full = (
            self.q_proj(x_norm)
            .view(B, S, self.num_q_heads, self.head_dim)
            .transpose(1, 2)
        )
        k_step = (
            self.k_proj(x_norm)
            .view(B, S, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v_step = (
            self.v_proj(x_norm)
            .view(B, S, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        if kv_cache is not None:
            k_full, v_full = kv_cache.update(k_step, v_step)
        else:
            k_full, v_full = k_step, v_step

        kv_indices, kv_weights, aux_loss, routing_entropy = self.router(
            k_full, window_size=self.window_size
        )

        K_len = kv_indices.size(-1)

        kv_gather_idx = kv_indices.unsqueeze(-1).expand(
            B, self.num_kv_heads, K_len, self.head_dim
        )
        k_sparse = torch.gather(k_full, dim=2, index=kv_gather_idx)
        v_sparse = torch.gather(v_full, dim=2, index=kv_gather_idx)

        # Handle Query position indexing for single decoding step vs. prompt evaluation
        if kv_cache is not None and S == 1:
            curr_pos = kv_cache.seq_pos - 1
            q_indices = torch.full((B, self.num_q_heads, 1), curr_pos, device=x.device, dtype=torch.long)
            q_sparse = q_full
            q_weights = torch.ones_like(q_indices, dtype=x.dtype)
        else:
            q_indices = kv_indices.repeat_interleave(self.num_q_per_kv, dim=1)
            q_weights = kv_weights.repeat_interleave(self.num_q_per_kv, dim=1)

            q_gather_idx = q_indices.unsqueeze(-1).expand(
                B, self.num_q_heads, K_len, self.head_dim
            )
            q_sparse = torch.gather(q_full, dim=2, index=q_gather_idx)

        q = apply_rotary_emb_indexed(q_sparse, cos, sin, q_indices)
        k = apply_rotary_emb_indexed(k_sparse, cos, sin, kv_indices)

        k = k.repeat_interleave(self.num_q_per_kv, dim=1)
        v = v_sparse.repeat_interleave(self.num_q_per_kv, dim=1)

        raw_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)

        kv_indices_q = kv_indices.repeat_interleave(self.num_q_per_kv, dim=1)
        causal_mask = q_indices.unsqueeze(-1) < kv_indices_q.unsqueeze(-2)

        masked_scores = raw_scores.masked_fill(causal_mask, float("-inf"))
        attention_weights = F.softmax(masked_scores, dim=-1)
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)

        attention_output = torch.matmul(attention_weights, v)

        if kv_cache is not None and S == 1:
            stitched = (
                attention_output.transpose(1, 2)
                .contiguous()
                .view(B, S, self.d_model)
            )
        else:
            weighted_output = attention_output * q_weights.unsqueeze(-1)
            output_full = torch.zeros_like(q_full)
            output_full.scatter_add_(dim=2, index=q_gather_idx, src=weighted_output)

            stitched = (
                output_full.transpose(1, 2)
                .contiguous()
                .view(B, S, self.d_model)
            )
            
        attn_out = self.out_proj(stitched)
        
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        
        return x, aux_loss, routing_entropy


@dataclass
class nanoSubQConfig:
    vocab_size: int = 50304
    max_seq_len: int = 2048
    d_model: int = 768
    num_layers: int = 12
    num_q_heads: int = 12
    num_kv_heads: int = 4
    window_size: int = 8
    dropout: float = 0.0


class nanoSubQ(nn.Module):
    def __init__(self, config: nanoSubQConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.d_model),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([
                SubQAttention(
                    d_model=config.d_model,
                    num_q_heads=config.num_q_heads,
                    num_kv_heads=config.num_kv_heads,
                    window_size=config.window_size,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]),
            ln_f = nn.LayerNorm(config.d_model),
        ))

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.head_dim = config.d_model // config.num_q_heads
        self.precompute_rope(config.max_seq_len, self.head_dim)

        self.apply(self._init_weights)

    def precompute_rope(self, max_seq_len: int, head_dim: int):
        positions = torch.arange(max_seq_len).unsqueeze(1)
        inv_freq = 1.0 / (
            10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        angles = positions * inv_freq

        self.register_buffer(
            "cos_cached", torch.cos(angles).unsqueeze(0).unsqueeze(0), persistent=False
        )
        self.register_buffer(
            "sin_cached", torch.sin(angles).unsqueeze(0).unsqueeze(0), persistent=False
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None, kv_caches: list[SparseKVCache] = None):
        device = idx.device
        b, t = idx.size()

        tok_emb = self.transformer.wte(idx)
        x = self.transformer.drop(tok_emb)

        total_aux_loss = torch.tensor(0.0, device=device, dtype=x.dtype)
        total_entropy = torch.tensor(0.0, device=device, dtype=x.dtype)

        for i, block in enumerate(self.transformer.h):
            cache = kv_caches[i] if kv_caches is not None else None
            x, aux_loss, routing_entropy = block(x, self.cos_cached, self.sin_cached, kv_cache=cache)
            total_aux_loss = total_aux_loss + aux_loss
            total_entropy = total_entropy + routing_entropy

        x = self.transformer.ln_f(x)
        avg_entropy = total_entropy / len(self.transformer.h)

        if targets is not None:
            logits = self.lm_head(x)
            task_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            loss = task_loss + total_aux_loss
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss, avg_entropy

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = None, use_cache: bool = True, generator: torch.Generator = None):
        if use_cache:
            batch_size = idx.size(0)
            kv_caches = [
                SparseKVCache(
                    batch_size=batch_size,
                    num_kv_heads=self.config.num_kv_heads,
                    max_seq_len=self.config.max_seq_len,
                    head_dim=self.head_dim,
                    device=idx.device,
                    dtype=self.transformer.wte.weight.dtype
                )
                for _ in range(self.config.num_layers)
            ]
            
            logits, _, _ = self(idx, kv_caches=kv_caches)
            
            for _ in range(max_new_tokens):
                step_logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(step_logits, min(top_k, step_logits.size(-1)))
                    step_logits[step_logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(step_logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1, generator=generator)
                idx = torch.cat((idx, idx_next), dim=1)
                
                logits, _, _ = self(idx_next, kv_caches=kv_caches)
        else:
            for _ in range(max_new_tokens):
                idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]
                logits, _, _ = self(idx_cond)
                step_logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(step_logits, min(top_k, step_logits.size(-1)))
                    step_logits[step_logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(step_logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1, generator=generator)
                idx = torch.cat((idx, idx_next), dim=1)

        return idx


# =============================================================================
# INTEGRATION & VERIFICATION TEST SUITE
# =============================================================================

def run_tests():
    print("Running verification tests...")

    config = nanoSubQConfig(
        vocab_size=1000,
        max_seq_len=128,
        d_model=64,
        num_layers=2,
        num_q_heads=4,
        num_kv_heads=2,
        window_size=8
    )
    model = nanoSubQ(config)

    # Test 1: Straight-Through Estimator Gradient Flow
    ste = StraightThroughEstimator(threshold=0.5)
    soft_inputs = torch.tensor([0.2, 0.7, 0.4, 0.9], requires_grad=True)
    hard_outputs = ste(soft_inputs)
    assert torch.equal(hard_outputs.detach(), torch.tensor([0.0, 1.0, 0.0, 1.0])), "STE forward output is incorrect"
    loss = hard_outputs.sum()
    loss.backward()
    assert soft_inputs.grad is not None and torch.equal(soft_inputs.grad, torch.ones_like(soft_inputs)), "STE gradient backprop failed"
    print("Test 1 Passed: Straight-Through Estimator forward & backward gradient flow verified.")

    # Test 2: PerKVHeadRouter Selection & Dimensions
    head_dim = config.d_model // config.num_q_heads
    router = PerKVHeadRouter(head_dim=head_dim, num_kv_heads=config.num_kv_heads)
    mock_k = torch.randn(2, config.num_kv_heads, 32, head_dim)
    indices, weights, aux_loss, entropy = router(mock_k, window_size=8)
    assert indices.shape[0] == 2 and indices.shape[1] == config.num_kv_heads, "Router index dimensions incorrect"
    assert weights.shape == indices.shape, "Router weights shape mismatch"
    assert aux_loss.item() >= 0.0, "Auxiliary loss should be non-negative"
    print("Test 2 Passed: PerKVHeadRouter dimensions and metrics verified.")

    # Test 3: Forward Pass & Target Loss Computation
    batch_size, seq_len = 2, 16
    dummy_input = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    dummy_targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    logits, loss, entropy = model(dummy_input, targets=dummy_targets)
    assert logits.shape == (batch_size, seq_len, config.vocab_size), f"Expected logits shape {(batch_size, seq_len, config.vocab_size)}, got {logits.shape}"
    assert loss is not None and loss.item() > 0, "Loss computation failed"
    print("Test 3 Passed: Full model forward pass & target loss verified.")

    # Test 4: Backward Pass across full nanoSubQ model
    model.zero_grad()
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} did not receive gradients"
    print("Test 4 Passed: Full backward pass succeeded across all parameters.")

    # Test 5: Inference Generation Loop
    prompt = torch.randint(0, config.vocab_size, (1, 8))
    generated = model.generate(prompt, max_new_tokens=5, temperature=1.0, top_k=10, use_cache=False)
    assert generated.shape == (1, 13), f"Expected output shape (1, 13), got {generated.shape}"
    print("Test 5 Passed: Autoregressive token generation verified.")

    # Test 6: Temperature Scheduler Step
    scheduler = TemperatureScheduler(model, t_max=2.0, t_min=0.1, total_steps=100)
    new_t = scheduler.step(current_step=50)
    assert new_t < 2.0 and new_t > 0.1, "Temperature annealing step failed"
    print("Test 6 Passed: TemperatureScheduler functioning properly.")

    # Test 7: Verify KV Token Pruning Ratio
    seq_len = 128
    window_size = 8
    router_test = PerKVHeadRouter(head_dim=16, num_kv_heads=2, retention_ratio=0.40)
    test_k = torch.randn(1, 2, seq_len, 16)
    indices, _, _, _ = router_test(test_k, window_size=window_size)
    retained_tokens = indices.size(-1)
    pruning_ratio = (1.0 - (retained_tokens / seq_len)) * 100
    assert 50.0 <= pruning_ratio <= 70.0, f"Pruning ratio {pruning_ratio:.2f}% outside expected target range!"
    print(f"Test 7 Passed: Retained {retained_tokens}/{seq_len} tokens. Pruning Ratio = {pruning_ratio:.2f}%")

    # Test 8: Pre-allocated SparseKVCache Generation Match
    prompt_cache = torch.randint(0, config.vocab_size, (1, 8))
    
    g_cached = torch.Generator(device=prompt_cache.device).manual_seed(42)
    gen_cached = model.generate(prompt_cache.clone(), max_new_tokens=4, temperature=0.1, use_cache=True, generator=g_cached)
    
    g_uncached = torch.Generator(device=prompt_cache.device).manual_seed(42)
    gen_uncached = model.generate(prompt_cache.clone(), max_new_tokens=4, temperature=0.1, use_cache=False, generator=g_uncached)
    
    assert torch.equal(gen_cached, gen_uncached), "Cached and uncached generation outputs do not match!"
    print("Test 8 Passed: Pre-allocated SparseKVCache generation matches standard decoding.")

    print("\nAll integration tests completed successfully!")

if __name__ == "__main__":
    run_tests()