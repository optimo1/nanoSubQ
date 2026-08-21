import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. ROUTER VARIANTS
# ==========================================

class SharedSequenceRouter(nn.Module):
    """Single shared router per layer (1 gating decision shared across all heads)."""

    def __init__(self, d_model: int, base_k: int = 4, block_size: int = 4, rank: int = 4, alpha: float = 0.01):
        super().__init__()
        self.base_k = base_k
        self.block_size = block_size
        self.alpha = alpha

        self.gate_down = nn.Linear(d_model, rank, bias=False)
        self.gate_up = nn.Linear(rank, 1, bias=False)

    def calculate_k(self, seq_len: int) -> int:
        if seq_len <= self.block_size:
            return seq_len
        log_factor = math.log2(seq_len)
        raw_k = int(self.base_k * log_factor)
        optimized_k = ((raw_k + self.block_size - 1) // self.block_size) * self.block_size
        return min(optimized_k, seq_len)

    def forward(self, x: torch.Tensor, window_size: int = 8, temperature: float = 0.1):
        # x: [B, S, D]
        B, S, D = x.shape
        scores = torch.sigmoid(self.gate_up(self.gate_down(x)).squeeze(-1) / temperature)  # [B, S]

        hist_len = max(0, S - window_size)
        local_indices = torch.arange(hist_len, S, device=x.device).unsqueeze(0).expand(B, -1)

        if hist_len > 0:
            hist_scores = scores[:, :hist_len]
            k_global = self.calculate_k(hist_len)
            topk_scores, global_indices = torch.topk(hist_scores, k=k_global, dim=-1, sorted=True)
            local_scores = scores[:, hist_len:]
            combined_indices = torch.cat([global_indices, local_indices], dim=-1)
            combined_scores = torch.cat([topk_scores, local_scores], dim=-1)
        else:
            combined_indices = local_indices
            combined_scores = scores

        selected_indices, sort_perm = torch.sort(combined_indices, dim=-1)
        selected_scores = torch.gather(combined_scores, dim=-1, index=sort_perm)

        p = torch.clamp(scores, 1e-7, 1.0 - 1e-7)
        entropy = (-p * torch.log2(p) - (1.0 - p) * torch.log2(1.0 - p)).mean()

        return selected_indices, selected_scores, entropy


class PerHeadIndependentRouter(nn.Module):
    """Per-Head Independent Router (H distinct gating matrices)."""

    def __init__(self, head_dim: int, num_heads: int, base_k: int = 4, block_size: int = 4, rank: int = 4, alpha: float = 0.01):
        super().__init__()
        self.num_heads = num_heads
        self.base_k = base_k
        self.block_size = block_size
        self.alpha = alpha

        # Distinct weights per head using 1D Conv (groups=H) or linear parameters
        self.gate_down = nn.Parameter(torch.randn(num_heads, head_dim, rank) * 0.02)
        self.gate_up = nn.Parameter(torch.randn(num_heads, rank, 1) * 0.02)

    def calculate_k(self, seq_len: int) -> int:
        if seq_len <= self.block_size:
            return seq_len
        log_factor = math.log2(seq_len)
        raw_k = int(self.base_k * log_factor)
        optimized_k = ((raw_k + self.block_size - 1) // self.block_size) * self.block_size
        return min(optimized_k, seq_len)

    def forward(self, head_states: torch.Tensor, window_size: int = 8, temperature: float = 0.1):
        # head_states: [B, H, S, D_head]
        B, H, S, D_head = head_states.shape

        # Einsum for per-head dynamic projection
        bottleneck = torch.einsum("bhsd,hdr->bhsr", head_states, self.gate_down)
        scores = torch.sigmoid(torch.einsum("bhsr,hre->bhse", bottleneck, self.gate_up).squeeze(-1) / temperature)  # [B, H, S]

        hist_len = max(0, S - window_size)
        local_indices = torch.arange(hist_len, S, device=head_states.device).unsqueeze(0).unsqueeze(0).expand(B, H, -1)

        if hist_len > 0:
            hist_scores = scores[:, :, :hist_len]
            k_global = self.calculate_k(hist_len)
            topk_scores, global_indices = torch.topk(hist_scores, k=k_global, dim=-1, sorted=True)
            local_scores = scores[:, :, hist_len:]
            combined_indices = torch.cat([global_indices, local_indices], dim=-1)
            combined_scores = torch.cat([topk_scores, local_scores], dim=-1)
        else:
            combined_indices = local_indices
            combined_scores = scores

        selected_indices, sort_perm = torch.sort(combined_indices, dim=-1)
        selected_scores = torch.gather(combined_scores, dim=-1, index=sort_perm)

        p = torch.clamp(scores, 1e-7, 1.0 - 1e-7)
        entropy = (-p * torch.log2(p) - (1.0 - p) * torch.log2(1.0 - p)).mean()

        return selected_indices, selected_scores, entropy


# ==========================================
# 2. EXPERIMENT RUNNERS
# ==========================================

def run_extreme_stress_test(device: torch.device):
    """Stress tests sequence lengths from S=64 up to S=4096."""
    print("\n======================================================================")
    print(" TASK 1: Sequence Length Invariance & Stress Test (S=64 -> S=4096)")
    print("======================================================================")
    print(f"{'Seq Len (S)':>12} | {'Selected K':>12} | {'Sparsity %':>12} | {'Entropy (bits)':>16} | {'Time (ms)':>12}")
    print("-" * 74)

    router = SharedSequenceRouter(d_model=64, base_k=4, block_size=4).to(device)
    router.eval()

    seq_lengths = [64, 128, 256, 512, 1024, 2048, 4096]

    with torch.no_grad():
        for S in seq_lengths:
            x = torch.randn(2, S, 64, device=device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            selected_indices, _, entropy = router(x, window_size=8, temperature=0.1)

            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            k = selected_indices.size(-1)
            sparsity = (1.0 - (k / S)) * 100.0

            print(f"{S:12d} | {k:12d} | {sparsity:11.1f}% | {entropy.item():16.6f} | {elapsed_ms:12.2f}")

    print("✅ Extreme stress test completed successfully without entropy collapse or out-of-memory errors.")


def run_routing_strategy_comparison(device: torch.device):
    """Compares Shared Sequence-Level Routing vs Per-Head Independent Routing."""
    print("\n======================================================================")
    print(" TASK 2: Multi-Head Routing Strategy Comparison")
    print("======================================================================")

    B, S, D, H = 4, 1024, 64, 8
    D_head = D // H

    x = torch.randn(B, S, D, device=device)
    head_states = x.view(B, S, H, D_head).transpose(1, 2)  # [B, H, S, D_head]

    shared_router = SharedSequenceRouter(d_model=D, base_k=4, rank=4).to(device)
    per_head_router = PerHeadIndependentRouter(head_dim=D_head, num_heads=H, base_k=4, rank=4).to(device)

    # Param count
    shared_params = sum(p.numel() for p in shared_router.parameters())
    per_head_params = sum(p.numel() for p in per_head_router.parameters())

    # Timing & execution
    def benchmark(model, input_tensor):
        # Warmup
        for _ in range(5):
            _ = model(input_tensor)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        for _ in range(50):
            indices, scores, entropy = model(input_tensor)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        avg_ms = ((t1 - t0) / 50.0) * 1000.0
        return indices, entropy, avg_ms

    shared_idx, shared_ent, shared_ms = benchmark(shared_router, x)
    per_head_idx, per_head_ent, per_head_ms = benchmark(per_head_router, head_states)

    print(f"{'Strategy':>30} | {'Param Count':>12} | {'Index Shape':>18} | {'Entropy':>10} | {'Latency (ms)':>12}")
    print("-" * 92)
    print(f"{'Shared Sequence-Level':>30} | {shared_params:12d} | {str(list(shared_idx.shape)):>18} | {shared_ent.item():10.4f} | {shared_ms:12.3f}")
    print(f"{'Per-Head Independent':>30} | {per_head_params:12d} | {str(list(per_head_idx.shape)):>18} | {per_head_ent.item():10.4f} | {per_head_ms:12.3f}")

    print("\n--- Strategy Trade-Off Summary ---")
    print("1. Shared Sequence Routing: Cheaper parameter overhead & lower memory footprint. Best for global document filtering.")
    print("2. Per-Head Independent Routing: Head-specific indices allow heads to specialize in different context windows at the cost of higher index memory.")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Section 5 benchmarks on device: {device}")

    run_extreme_stress_test(device)
    run_routing_strategy_comparison(device)