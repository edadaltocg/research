import logging
import math
import random
import time
from itertools import chain
from typing import Any

import numpy as np
import torch
import torch.utils.benchmark as benchmark
from PIL import Image
from torch import Tensor, nn
from tqdm import tqdm

log = logging.getLogger(__name__)


def dummy_image(w, h):
    return Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))


def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
    t0 = benchmark.Timer(stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f})
    return t0.blocked_autorange().mean * 1e6


def seed_all(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.mps.manual_seed(seed)
    random.seed(seed)


def get_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def num_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class LoadPreTrainedModelWithLowMemoryContext:
    """Load a pre-trained model with low memory usage.

    Example:
        ```python
        with LoadPreTrainedModelWithLowMemoryContext(
            path / "dummy_model_original.pth",
            torch.device("cpu"),
            torch.float32,
        ) as ctx:
            model = DummyModel()
            ctx.load_state_dict(model)
        ```
    """

    def __init__(
        self,
        state_dict_path,
        target_device=None,
        target_dtype=torch.float32,
        inception_device=None,
    ) -> None:
        self.state_dict_path = state_dict_path
        self.target_device = target_device if target_device is not None else torch.device("cpu")
        self.target_dtype = target_dtype
        self.inception_device = inception_device if inception_device is not None else torch.device("meta")

    def __enter__(self):
        w_mmaped = torch.load(str(self.state_dict_path), map_location=self.target_device, mmap=True)
        log.info(f"Loading state_dict to memory with {self.target_device=} and {self.target_dtype=}")
        self.state_dict = {}
        total_mem = 0
        t0 = time.time()
        for k, v in tqdm(w_mmaped.items(), desc="Loading"):
            v = v.to(self.target_device, self.target_dtype)
            mem = v.element_size() * v.numel()
            total_mem += mem
            log.info(f"Key: {k}, Storage: {mem:,.2f} bytes")
            self.state_dict[k] = v
            del v
        t1 = time.time()
        log.info(f"Loading time: {t1 - t0:.2f} seconds")
        log.info(f"Memory taken by state_dict: {total_mem:,.2f} bytes")

        k0 = next(iter(self.state_dict.keys()))
        assert isinstance(self.state_dict, dict)
        assert isinstance(self.state_dict[k0], Tensor)
        assert self.state_dict[k0].device == self.target_device
        assert self.state_dict[k0].dtype == self.target_dtype
        torch.set_default_device(self.inception_device)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        torch.set_default_device("cpu")
        del self.state_dict

    def load_state_dict(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if k not in self.state_dict:
                log.warning(f"Key {k} not found in state_dict, copying to state_dict")
                if v.device.type == "meta":
                    self.state_dict[k] = torch.randn(v.shape, device=self.target_device, dtype=self.target_dtype)
                else:
                    self.state_dict[k] = v.clone().to(self.target_device, self.target_dtype)
        for submodule_name, submodule in model.named_modules():
            for param_name, param in list(
                chain(submodule.named_parameters(recurse=False), submodule.named_buffers(recurse=False))
            ):
                if len(param_name.split(".")) == 1:
                    # is leaf module
                    key = f"{submodule_name}{'.' if len(submodule_name) else ''}{param_name}"
                    try:
                        new_val = torch.nn.Parameter(self.state_dict[key].clone(), requires_grad=False)
                        setattr(submodule, param_name, new_val)
                    except KeyError:
                        log.warning(f"Key {key} not found in state_dict, creating new random tensor")
                        new_val = torch.nn.Parameter(
                            torch.randn(
                                param.shape,
                                device=self.target_device,
                                dtype=self.target_dtype,
                            ),
                            requires_grad=False,
                        )
                        setattr(submodule, param_name, new_val)
                        continue
                    finally:
                        if key in self.state_dict:
                            del self.state_dict[key]


def num_parameters(module: nn.Module, requires_grad: bool | None = None) -> int:
    total = 0
    for p in module.parameters():
        if requires_grad is None or p.requires_grad == requires_grad:
            if hasattr(p, "quant_state"):
                # bitsandbytes 4bit layer support
                quant_state: Any = p.quant_state
                total += math.prod(quant_state[1])
            else:
                total += p.numel()
    return total
