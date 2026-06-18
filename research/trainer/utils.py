from itertools import repeat
from typing import Any

import torch


def repeater(data_loader):
    for loader in repeat(data_loader):
        yield from loader


def move_to_device(o: Any, device: torch.device) -> Any:
    if isinstance(o, torch.Tensor):
        return o.to(device)
    if isinstance(o, dict):
        return {k: move_to_device(v, device) for k, v in o.items()}
    if isinstance(o, list):
        return [move_to_device(v, device) for v in o]
    if isinstance(o, tuple):
        return tuple(move_to_device(v, device) for v in o)
    return o
