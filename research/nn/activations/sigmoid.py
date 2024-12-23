import torch
from torch import Tensor


def sigmoid_math(x: Tensor):
    r"""
    $$
    f(X) = \frac{1}{1+e^{-X}}
    $$

    Args:
        x: `Tensor[n,1]`
    """
    return 1 / (1 + torch.exp(-x))


def sigmoid_math_derivative(x: Tensor):
    return (1 - sigmoid_math(x)) * sigmoid_math(x)
