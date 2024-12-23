from torch import Tensor


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
