from torch import Tensor

from research.nn.activations.softmax import log_softmax_math


def nl(softmax: Tensor, target: list[int] | Tensor) -> Tensor:
    return -softmax[:, target].log().mean()


def nll(log_softmax: Tensor, target: list[int] | Tensor) -> Tensor:
    return -log_softmax[:, target].mean()


def ce(logits: Tensor, target: list[int] | Tensor) -> Tensor:
    return nll(log_softmax_math(logits), target)
