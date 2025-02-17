import torch
from torch import Tensor, nn


def sigmoid_math(x: Tensor):
    r"""
    $$
    f(X) = \frac{1}{1+e^{-X}}
    $$

    Args:
        x: `Tensor[n,1]`
    """
    return 1 / (1 + torch.exp(-x))


class SigmoidMath(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return sigmoid_math(x)


def sigmoid_math_derivative(x: Tensor):
    return (1 - sigmoid_math(x)) * sigmoid_math(x)


class SigmoidMathDerivative(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return sigmoid_math_derivative(x)


def stable_sigmoid(x: Tensor):
    r"""
    Numerically stable sigmoid function.

    Args:
        x: `Tensor[n,1]`
    """
    # Use different formulas based on the sign of x
    positive_mask = x >= 0
    negative_mask = ~positive_mask

    result = torch.zeros_like(x)

    # For positive x, use the standard sigmoid formula
    result[positive_mask] = 1 / (1 + torch.exp(-x[positive_mask]))

    # For negative x, use the alternative formula
    exp_x = torch.exp(x[negative_mask])
    result[negative_mask] = exp_x / (1 + exp_x)

    return result


class StableSigmoid(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return stable_sigmoid(x)
