
import torch
import torch.nn.functional as F
from torch import nn


class MoELayer(nn.Module):
    def __init__(
        self, experts: list[nn.Module], gate: nn.Module, num_experts_per_tok: int = 2
    ):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.num_experts_per_tok = num_experts_per_tok

    def forward(self, inputs: torch.Tensor):
        gate_logits = self.gate(inputs)
        weights, selected_experts = torch.topk(gate_logits, self.num_experts_per_tok)
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)
        results = torch.zeros_like(inputs)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(
                inputs[batch_idx]
            )
        return results
