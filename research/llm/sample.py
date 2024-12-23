from typing import Optional

import torch
from torch import Tensor


def greedy_sample(logits: Tensor):
    return torch.argmax(logits[:, -1], dim=-1)


def top_p_probs(probs: Tensor, p: float):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    return probs


def top_k_logits(logits: Tensor, top_k: int):
    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    # select the very last value from the top_k above as the pivot
    pivot = v.select(-1, -1).unsqueeze(-1)
    # set everything smaller than pivot value to inf since these
    # should be pruned
    logits = torch.where(logits < pivot, -float("Inf"), logits)
    return logits


def sample(
    logits: Tensor,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> torch.Tensor:
    # scale the logits based on temperature
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        logits = top_k_logits(logits, top_k)

    # change logits into probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)

    if top_p is not None:
        probs = top_p_probs(probs, top_p)

    # multinomial sample
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)
