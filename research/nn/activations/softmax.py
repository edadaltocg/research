from torch import Tensor, nn


def softmax_math(x: Tensor):
    r"""
    $$
    f(x)_i = \frac{e^(x_i)}{\sum_i e^(x_i)}
    $$

    Args:
        x: Tensor[n,k]
    """
    return x.exp() / x.exp().sum(dim=-1, keepdim=True)


def log_softmax_math(x: Tensor):
    r"""
    $$
    f(x)_i = x - \log(\sum_i e^(x_i))
    $$

    Args:
        x: Tensor[n,k]
    """
    return x - x.exp().sum(dim=-1, keepdim=True).log()


def stable_softmax(x: Tensor):
    r"""
    Numerically stable softmax function.
    $$
    f(x)_i = \frac{e^{x_i - \max(x)}}{\sum_i e^{x_i - \max(x)}}
    $$

    Args:
        x: Tensor[n,k]
    Returns:
        A tensor with the softmax probabilities.
    """
    x_max, _ = x.max(dim=-1, keepdim=True)
    x_exp = (x - x_max).exp()
    return x_exp / x_exp.sum(dim=-1, keepdim=True)


def stable_log_softmax(x: Tensor):
    r"""
    Numerically stable log softmax function.
    $$
    f(x)_i = x_i - \max(x) - \log(\sum_i e^{x_i - \max(x)})
    $$

    Args:
        x: Tensor[n,k]
    Returns:
        A tensor with the log softmax values.
    """
    x_max, _ = x.max(dim=-1, keepdim=True)
    return x - x_max - (x - x_max).exp().sum(dim=-1, keepdim=True).log()


class SoftmaxMath(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return softmax_math(x)


class LogSoftmaxMath(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return log_softmax_math(x)


class StableSoftmax(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return stable_softmax(x)


class StableLogSoftmax(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return stable_log_softmax(x)
