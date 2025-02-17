from functools import partial

from torch import Tensor, nn


import logging

log = logging.getLogger(__file__)


def init_weights(m, init_fn: str, **kwargs) -> None:
    """
    Initialize weights of a given module `m` using the specified initialization function `init_fn`.

    Parameters:
        m (torch.nn.Module): The module whose weights are to be initialized.
        init_fn (str): A string specifying the initialization function. Options are "normal", "kaiming_normal", and "uniform".
        kwargs: Additional keyword arguments to pass to the initialization function.

    Raises:
        ValueError: If `init_fn` is not one of the specified options.
    """

    fn = lambda x: x
    if init_fn == "normal":
        fn = partial(nn.init.normal_, **kwargs)
    elif init_fn == "uniform":
        fn = partial(nn.init.uniform_, **kwargs)
    elif init_fn == "kaiming_uniform":
        fn = partial(nn.init.kaiming_uniform_, **kwargs)
    elif init_fn == "kaiming_normal":
        fn = partial(nn.init.kaiming_normal_, **kwargs)
    elif init_fn == "xavier_uniform":
        fn = partial(nn.init.xavier_uniform_, **kwargs)
    elif init_fn == "xavier_normal":
        fn = partial(nn.init.xavier_normal_, **kwargs)
    elif init_fn == "default":
        fn = lambda x: x.reset_parameters()
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
        return
    else:
        raise NotImplementedError(f"Initialization function '{init_fn}' is not implemented.")

    # Initialize weights
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        fn(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
