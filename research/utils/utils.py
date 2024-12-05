import logging
import math
import os
import random
import time
from contextlib import suppress
from datetime import datetime, timedelta
from itertools import chain

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.benchmark as benchmark
from PIL import Image
from torch import Tensor, nn, optim
from tqdm import tqdm

log = logging.getLogger(__name__)


def dummy_image(w, h):
    return Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))


def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    return t0.blocked_autorange().mean * 1e6


def seed_all(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def num_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def distrib_setup(backend="nccl", master_addr="localhost", master_port="12355"):
    """Run with torchrun:

    Single node:
        torchrun
            --standalone
            --nnodes=1
            --nproc-per-node=$NUM_TRAINERS
            YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)
    """
    master_addr = os.environ.get("MASTER_ADDR", master_addr)
    master_port = os.environ.get("MASTER_PORT", master_port)

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    # initialize the process group
    dist.init_process_group(backend, timeout=timedelta(seconds=1800))


def distrib_cleanup():
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


def distrib_spawn(fn, *args):
    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(fn, args=(WORLD_SIZE, args), nprocs=WORLD_SIZE, join=True)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process():
    return get_rank() == 0


def print_rank0(msg):
    if is_main_process():
        print(msg)


def clear_gpu_cache(rank=None):
    """Clear the GPU cache for all ranks"""
    if rank == 0:
        print("Clearing GPU cache for all ranks")
    torch.cuda.empty_cache()


def print_model_size(model, config, rank: int = 0) -> None:
    """
    Print model name, the number of trainable parameters and initialization time.

    Args:
        model: The PyTorch model.
        model_name (str): Name of the model.
        init_time_start (float): Initialization start time.
        init_time_end (float): Initialization end time.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        print(f"--> Model {config.model_name}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> {config.model_name} has {total_params / 1e6} Million params\n")


def save_model(ddp_model, path):
    if is_main_process():
        torch.save(ddp_model.state_dict(), path)
    if is_dist_avail_and_initialized():
        dist.barrier()


def save_checkpoint(checkpoint, path):
    if is_main_process():
        torch.save(checkpoint, path)
    if is_dist_avail_and_initialized():
        dist.barrier()


def reduce_across_processes(val, rank):
    if not is_dist_avail_and_initialized():
        if isinstance(val, Tensor):
            return val
        return torch.tensor(val)

    if not isinstance(val, Tensor):
        val = torch.tensor(val, device=rank)
    dist.barrier()
    dist.all_reduce(val)
    return val


def get_autocast(precision):
    if precision == "amp":
        return torch.cuda.amp.autocast
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return suppress


def get_input_dtype(precision: str):
    input_dtype = None
    if precision in ("bf16", "pure_bf16"):
        input_dtype = torch.bfloat16
    elif precision in ("fp16", "pure_fp16"):
        input_dtype = torch.float16
    return input_dtype


def get_date():
    """Create date and time for file save uniqueness

    Example:
        2022-05-07-08:31:12_PM'
    """
    date = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    return date


def format_to_gb(item):
    """Format numbers to gigabyte and round to 4 digit precision"""
    metric_num = item / 1024**3
    metric_num = round(metric_num, ndigits=4)
    return metric_num


def human_bytes(num, suffix="B"):
    """Convert bytes to human readable format"""
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f} Yi{suffix}"


def is_bf16_ready() -> bool:
    """Available on Ampere GPUs and later"""
    return (
        torch.version.cuda
        and torch.cuda.is_bf16_supported()
        and LooseVersion(torch.version.cuda) >= "11.0"
        and dist.is_nccl_available()
    )


def timed_cuda(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000


def get_criterion_cls(criterion_name: str) -> nn.modules.loss._Loss:
    return getattr(nn, criterion_name)


def get_optimizer_cls(optimizer_name: str) -> optim.Optimizer:
    return getattr(optim, optimizer_name)


def get_scheduler_cls(scheduler_name: str) -> optim.lr_scheduler._LRScheduler:
    return getattr(optim.lr_scheduler, scheduler_name)


def flatten_list_of_lists(l):
    return list(chain.from_iterable(l))


def collate_flat(batch):
    flattened = list(chain.from_iterable(batch))
    return flattened


def peak_gpu_memory(reset: bool = False) -> float | None:
    """
    Get the peak GPU memory usage in MB across all ranks.
    Only rank 0 will get the final result.
    """
    if not torch.cuda.is_available():
        return None

    device = torch.device("cuda")
    peak_mb = torch.cuda.max_memory_allocated(device) / 1000000
    if is_distributed():
        peak_mb_tensor = torch.tensor(peak_mb, device=device)
        dist.reduce(peak_mb_tensor, 0, dist.ReduceOp.MAX)
        peak_mb = peak_mb_tensor.item()

    if reset:
        # Reset peak stats.
        torch.cuda.reset_max_memory_allocated(device)

    return peak_mb


def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000


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
        target_device=torch.device("cpu"),
        target_dtype=torch.float32,
        inception_device=torch.device("meta"),
    ) -> None:
        self.state_dict_path = state_dict_path
        self.target_device = target_device
        self.target_dtype = target_dtype
        self.inception_device = inception_device

    def __enter__(self):
        w_mmaped = torch.load(
            str(self.state_dict_path), map_location=self.target_device, mmap=True
        )
        log.info(
            f"Loading state_dict to memory with {self.target_device=} and {self.target_dtype=}"
        )
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
        log.info(f"Memory taken by state_dict: {human_bytes(total_mem)}")

        k0 = list(self.state_dict.keys())[0]
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
                self.state_dict[k] = v.clone().to(self.target_device, self.target_dtype)
        for submodule_name, submodule in model.named_modules():
            for param_name, param in chain(
                submodule.named_parameters(), submodule.named_buffers()
            ):
                if len(param_name.split(".")) == 1:
                    # is leaf module
                    key = f"{submodule_name}{'.' if len(submodule_name) else ''}{param_name}"
                    try:
                        new_val = torch.nn.Parameter(
                            self.state_dict[key].clone(), requires_grad=False
                        )
                        setattr(submodule, param_name, new_val)
                    except KeyError:
                        log.warning(
                            f"Key {key} not found in state_dict, creating new random tensor"
                        )
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
                total += math.prod(p.quant_state[1])
            else:
                total += p.numel()
    return total


def flops_per_param(
    max_seq_length: int, n_layer: int, n_embd: int, n_params: int
) -> int:
    flops_per_token = (
        2 * n_params
    )  # each parameter is used for a MAC (2 FLOPS) per network operation
    # this assumes that all samples have a fixed length equal to the block size
    # which is most likely false during finetuning
    flops_per_seq = flops_per_token * max_seq_length
    attn_flops_per_seq = n_layer * 2 * 2 * (n_embd * (max_seq_length**2))
    return flops_per_seq + attn_flops_per_seq


def estimate_flops(model: nn.Module, training: bool) -> int:
    """Measures estimated FLOPs for MFU.

    Refs:
        * https://ar5iv.labs.arxiv.org/html/2205.05198#A1
        * https://ar5iv.labs.arxiv.org/html/2204.02311#A2
    """
    # using all parameters for this is a naive over estimation because not all model parameters actually contribute to
    # this FLOP computation (e.g. embedding, norm). For this reason, the result will be higher by a fixed percentage
    # (~10%) compared to the measured FLOPs, making those lower but more realistic.
    # For a proper estimate, this needs a more fine-grained calculation as in Appendix A of the paper.
    n_trainable_params = num_parameters(model, requires_grad=True)
    trainable_flops = flops_per_param(
        model.max_seq_length,
        model.config.n_layer,
        model.config.n_embd,
        n_trainable_params,
    )
    # forward + backward + gradients (assumes no gradient accumulation)
    ops_per_step = 3 if training else 1
    n_frozen_params = num_parameters(model, requires_grad=False)
    frozen_flops = flops_per_param(
        model.max_seq_length, model.config.n_layer, model.config.n_embd, n_frozen_params
    )
    # forward + backward
    frozen_ops_per_step = 2 if training else 1
    return ops_per_step * trainable_flops + frozen_ops_per_step * frozen_flops
