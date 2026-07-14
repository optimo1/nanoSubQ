#topk for relevancy
import math
import torch
import torch.nn as nn

'''
x = torch.tensor([-1.0, 5.8, 3.0, -3.1, -3.2])

values, _ = torch.topk(x, k=1)
treshold = values[-1]

print(values)
print(treshold)

binary_mask = (x >= treshold).float()

print(binary_mask)
'''


class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, raw_scores: torch.Tensor, hard_mask: torch.Tensor) -> torch.Tensor:
        ctx.saved_for_backward(raw_scores)
        return hard_mask

    @staticmethod
    def backward(ctx, *grad_output: torch.Tensor):
        in_grad = grad_output[0]

        raw_scores, = ctx.saved_tensors
        modified_grad = in_grad * (torch.sigmoid(raw_scores))

        return modified_grad, None

class Router(nn.Module):
    def __init__(self, n_embd: int, base_k: int = 32, block_size: int = 32):
        super().__init__()

        self.base_k = base_k
        self.block_size = block_size
        self.gate_proj = nn.Linear(n_embd, 1)

    def calculate_k(self, seq_len: int) -> int:
        if seq_len <= self.block_size:
            return seq_len

        log_factor = math.log2(seq_len)
        raw_k = int(self.base_k * log_factor)
        optimized_k = ((raw_k + self.block_size - 1)// self.block_size) * self.block_size
        return min(optimized_k, seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        S, W, C = x.size()

        raw_scores = self.gate_proj(x).unsqueeze(-1)
        k = self.calculate_k(W)

        topk_values, _ = torch.topk(raw_scores, k=k, dim=-1)
        treshold = topk_values[..., -1:]

        hard_mask = (raw_scores >= treshold).float()
        final_gate = STE.apply(raw_scores, hard_mask)

        assert isinstance(final_gate, torch.Tensor)
        return final_gate.unsqueeze(-1)