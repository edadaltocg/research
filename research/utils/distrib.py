import logging
import os
from datetime import timedelta
from typing import Tuple

import torch
import torch.distributed as dist

log = logging.getLogger(__name__)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


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


def get_world_size_and_rank() -> Tuple[int, int]:
    """Function that gets the current world size (aka total number
    of ranks) and rank number of the current process in the default process group.

    Returns:
        Tuple[int, int]: world size, rank
    """
    if dist.is_available() and dist.is_initialized():
        return torch.distributed.get_world_size(), torch.distributed.get_rank()
    else:
        return 1, 0


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


def log_rank_zero(logger: logging.Logger, msg: str, level: int = logging.INFO) -> None:
    """
    Logs a message only on rank zero.

    Args:
        logger (logging.Logger): The logger.
        msg (str): The warning message.
        level (int): The logging level. See https://docs.python.org/3/library/logging.html#levels for values.
            Defaults to ``logging.INFO``.
    """
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    if rank != 0:
        return
    logger.log(level, msg, stacklevel=2)
