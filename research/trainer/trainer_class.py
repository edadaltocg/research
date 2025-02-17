import logging
import math
import os
import random
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from itertools import repeat
from pathlib import Path
from typing import (
    Any,
    Literal,
    TypeVar,
)

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.utils.data
import tqdm
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullOptimStateDictConfig,
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import ProfilerActivity, schedule
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from torcheval.metrics import Max, Mean, Throughput
from torcheval.metrics.toolkit import sync_and_compute


T = TypeVar("T")

log = logging.getLogger(__name__)

"""
Utils
"""


@dataclass
class Trainer:
    """
    High performance Trainer class for research.
    Readable, minimalist, modularized, and flexible
    Based on pure PyTorch.
    Supports 1 -> N GPUs from 1 -> M nodes.

    Features:
        - LR Scheduling
        - FSDP
        - Mixed precision
        - Automatic gradient accumulation
        - Gradient clipping
        - CPUOffloading
        - Checkpointing
            - Model
            - Optimizers
            - General training state
        - Autorestart from last checkpoint
        - Clock aware checkpoint (useful when running timed jobs)
        - Backward prefetching
        - Speed monitor

    Example:

    References:
        [1] https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
        [2] https://pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html
        [3] https://github.com/allenai/OLMo/
    """

    # essentials
    model: nn.Module
    train_dataset: Dataset | IterableDataset
    optimizer: optim.Optimizer
    get_loss_and_metrics: Callable
    eval_dataset: Dataset | IterableDataset | None = None

    # dataloader config
    num_workers: int = 0
    drop_last: bool = False
    pin_memory: bool = False
    prefetch_factor: int | None = None
    persistent_workers: bool = False
    device_train_batch_size: int = 1
    device_eval_batch_size: int | None = None

    debug_run: bool = False

    num_steps: int = 1
    start_step: int = 0
    grad_accumulation_steps: int = 1

    num_logs: int = 1000
    log_format: str = "[%(levelname)s] %(message)s"
    log_filename: str = "stdout.log"
    log_level: int = logging.INFO

    num_checkpoints: int = 5
    checkpoint_best: bool = True
    checkpoint_last: bool = True
    checkpoint_prefix: str = "checkpoint"

    num_evals: int = 10
    eval_on_start: bool = True
    eval_on_end: bool = True

    resume_from_checkpoint: bool = True
    checkpoint_sharded: bool = False
    checkpoint_path: str = "auto"

    run_name: str = ""
    output_root: str = "output/train"
    seed: int = 42

    # advanced training args
    max_grad_norm: float | None = None
    profile: bool = False
    force_cpu: bool = False

    ## control gradient accumulation
    effective_device_train_batch_size: int | None = None

    ## compiler args
    compile: bool = True
    compiler_mode: str | None = None
    compiler_fullgraph: bool = False
    compiler_backend: str = "inductor"

    ## fsdp (if not, try DDP or standard)
    fsdp: bool = False
    use_orig_params: bool = True
    sharding_strategy: ShardingStrategy = (
        ShardingStrategy.SHARD_GRAD_OP  # Zero2
    )  ### model parameters precision
    param_dtype: Literal["float16", "bfloat16", "float32"] = "float32"
    ### gradient communication precision.
    reduce_dtype: Literal["float16", "bfloat16", "float32"] = "float32"
    ### buffer precision.
    buffer_dtype: Literal["float16", "bfloat16", "float32"] = "float32"

    # profiling

    # optionals
    lr_scheduler: Any | None = None
    max_duration_in_hours: float | None = None

    def __post_init__(self):
        self.seed_all(self.seed)
        self.setup_output_folder()
        self.setup_logging()

        if not self.force_cpu:
            self.setup_distrib_if_available()

        self.device = (
            torch.device(
                f"cuda:{self.local_rank}"
                if torch.cuda.is_available() and torch.cuda.device_count() >= self.world_size
                else "cpu"
            )
            if not self.force_cpu
            else torch.device("cpu")
        )

        self.setup_model()
        self.setup_train_dataloader()
        if self.eval_dataset is not None:
            self.setup_eval_dataloader()

        if self.debug_run:
            self.num_steps = 1

        self.curr_step = self.start_step
        self.eval_every = math.ceil(self.num_steps / max(1, self.num_evals))
        self.log_every = math.ceil(self.num_steps / max(1, self.num_logs))
        self.checkpoint_every = math.ceil(self.num_steps / max(1, self.num_checkpoints))

        self.best_eval_loss: float = float("inf")

        self.metrics = {}
        self.metrics["train/throughput"] = Throughput(device=self.device)
        self.metrics["eval/throughput"] = Throughput(device=self.device)
        self.metrics["train/throughput"] = Throughput(device=self.device)
        self.metrics["peak_gpu_memory"] = Max(device=self.device)
        self.metrics["max_grad_norm"] = Max(device=self.device)
        self.results = defaultdict(list)

        # profiling
        self.profiling_schedule = schedule(wait=1, warmup=5, active=3, repeat=1)
        self.torch_profiler = torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False,
            with_stack=True,
            schedule=self.profiling_schedule,
            # on_trace_ready=on_trace_ready,
        )

        # TODO:load from checkpoint

        self.save_cfg_to_json()
        self.save_cfg_to_yaml()
        if self.local_rank == 0:
            self.log.info("Trainer initialized")
            self.log.info(self)

    @staticmethod
    def seed_all(seed: int = 42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)

    @property
    def _device_eval_batch_size(self):
        if self.device_eval_batch_size is None:
            self.log.warn("Setting device_eval_batch_size to device_train_batch_size")
            self.device_eval_batch_size = self.device_train_batch_size
        return self.device_eval_batch_size

    def setup_model(self):
        if self.fsdp and self.is_dist_avail_and_initialized() and self.device.type != "cpu":
            self.model = FSDP(
                self.model,
                # cpu_offload=CPUOffload.AUTO,
                backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # can increase the training speed in exchange for higher mem
                sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
                # mixed_precision=self.mixed_precision_policy,
                device_id=self.local_rank,
                # limit_all_gathers=True,
                use_orig_params=True,
            )
            # self.optimizer = optim.AdamW(
            #     self.model.parameters(), betas=(0.9, 0.999), lr=1e-3, weight_decay=1e-2
            # )
        elif self.is_dist_avail_and_initialized() and self.device.type != "cpu":
            self.model = self.model.to(self.device)
            self.model = DDP(self.model, [self.device])
        else:
            self.model = self.model.to(self.device)

        # compile
        if self.compile:
            self.model = torch.compile(
                self.model,
                mode=self.compiler_mode,
                backend=self.compiler_backend,
                fullgraph=self.compiler_fullgraph,
            )

    @staticmethod
    def setup_distrib_if_available(backend="nccl", master_addr="localhost", master_port="12355"):
        """Run with torchrun:

        Single node:
            torchrun
                --standalone
                --nnodes=1
                --nproc_per_node=$NUM_TRAINERS
                YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)
        """
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            os.environ["MASTER_ADDR"] = master_addr
            os.environ["MASTER_PORT"] = master_port
            rank = os.environ.get("LOCAL_RANK", None)
            if rank is None:
                log.warn("Consider running with torchrun.")
                return

            # initialize the process group
            dist.init_process_group(backend, timeout=timedelta(seconds=1800))

    def setup_output_folder(self):
        self.dest_path = Path(self.output_root) / self.run_name
        self.dest_path.mkdir(parents=True, exist_ok=True)

    """
    Logging
    """

    def setup_logging(self):
        # log in stdout and file
        file_handler = logging.FileHandler(filename=self.dest_path / self.log_filename)
        stdout_handler = logging.StreamHandler(stream=sys.stdout)
        if self.local_rank == 0:
            level = self.log_level
        else:
            level = logging.ERROR
        handlers = [file_handler, stdout_handler]
        logging.basicConfig(handlers=handlers, level=level, format=self.log_format)
        self.log = log

    """
    FSDP
    """

    @staticmethod
    def precision_validation(precision: Literal["float16", "bfloat16", "float32"]):
        v = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return v[precision]

    @property
    def mixed_precision_policy(self):
        MixedPrecision(
            param_dtype=self.precision_validation(self.param_dtype),
            reduce_dtype=self.precision_validation(self.reduce_dtype),
            buffer_dtype=self.precision_validation(self.buffer_dtype),
        )

    """
    Checkpointing
    """

    @property
    def model_state_dict(self):
        if self.fsdp and self.is_dist_avail_and_initialized() and False:
            with FSDP.state_dict_type(
                self.model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            ):
                return self.model.state_dict()
        return self.model.state_dict()

    @property
    def optimizer_state_dict(self):
        if self.fsdp and self.is_dist_avail_and_initialized() and False:
            with FSDP.state_dict_type(self.optimizer, StateDictType.FULL_STATE_DICT, self.fsdp_optim_policy):
                return self.optimizer.state_dict()
        return self.optimizer.state_dict()

    @property
    def train_loss(self):
        if "train/loss" not in self.results:
            return float("inf")
        return self.results["train/loss"][-1]

    def eval_loss(self):
        if "eval/loss" not in self.results:
            return float("inf")
        return self.results["eval/loss"][-1]

    """
    Distributed
    """

    def close(self):
        if self.is_dist_avail_and_initialized():
            dist.barrier()
            dist.destroy_process_group()

    @property
    def local_rank(self) -> int:
        if not self.is_dist_avail_and_initialized():
            return 0
        return int(os.environ["LOCAL_RANK"])

    @property
    def fsdp_optim_policy(self):
        return FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)

    @staticmethod
    def is_dist_avail_and_initialized():
        if not dist.is_available():
            return False
        if not dist.is_initialized():
            return False
        return True

    @property
    def world_size(self) -> int:
        if not Trainer.is_dist_avail_and_initialized():
            return 1
        return dist.get_world_size()

    def load_trainer_state_dict(self, state_dict: dict[str, Any]):
        raise NotImplementedError

    def restore_rng_state(self, rng_state: dict[str, Any]) -> None:
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch"])
        torch.cuda.set_rng_state(rng_state["cuda"])

    def restore_checkpoint(self):
        raise NotImplementedError

    def system_metrics(self) -> float:
        """
        Get the peak GPU memory usage in MB across all ranks.
        Only rank 0 will get the final result.
        """
        if not torch.cuda.is_available():
            return 0

        peak_mb = torch.cuda.max_memory_allocated(self.device) / 1000000
        if self.is_dist_avail_and_initialized():
            peak_mb_tensor = torch.tensor(peak_mb, device=self.device)
            dist.reduce(peak_mb_tensor, 0, dist.ReduceOp.MAX)
            peak_mb = peak_mb_tensor.item()

        # Reset peak stats.
        torch.cuda.reset_max_memory_allocated(self.device)

        return peak_mb

    @staticmethod
    def format_float(value: float) -> str:
        if value < 0.0001:
            return str(value)  # scientific notation
        elif value > 1000:
            return f"{int(value):,d}"
        elif value > 100:
            return f"{value:.1f}"
        elif value > 10:
            return f"{value:.2f}"
        elif value > 1:
            return f"{value:.3f}"
        else:
            return f"{value:.4f}"

    def log_metrics_to_console(self, prefix: str, metrics: dict[str, float]):
        print(f"{self.local_rank=}")

        self.log.info(
            f"*{prefix.upper()}* [{self.curr_step}/{self.num_steps}]\n"
            + "\t".join([
                f"    {name}={self.format_float(value)}"
                for name, value in metrics.items()
                if not name.startswith("optim/")  # there's too many optimizer metrics
            ])
        )

    def save_checkpoint(self, prefix: str | None = None):
        if self.is_dist_avail_and_initialized():
            # dist.barrier()
            pass
        if prefix is None:
            prefix = str(self.curr_step)
        print("SAVING CHECKPOINT", prefix)
        cuda_rng = ""
        if self.device.type == "cuda":
            cuda_rng = torch.cuda.get_rng_state()
        if False:
            with FSDP.state_dict_type(
                self.model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            ):
                model_state_dict = self.model.state_dict()

        trainer_state_dict = {
            "total_steps": self.num_steps,
            "curr_step": self.curr_step,
            "world_size": self.world_size,
            "model": self.model_state_dict,
            "optimizer": self.optimizer_state_dict,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.random.get_rng_state(),
                "cuda": cuda_rng,
            },
        }
        if self.local_rank == 0:
            torch.save(trainer_state_dict, self.dest_path / f"checkpoint_{prefix}.pt")
            print("SAVED CHECKPOINT", prefix)

    def save_model(self, prefix: str = "last"):
        print("SAVING MODEL", prefix)
        if self.is_dist_avail_and_initialized():
            # dist.barrier()
            pass
        model_state_dict = self.model_state_dict
        if self.local_rank == 0:
            torch.save(model_state_dict, self.dest_path / f"{prefix}_model.pt")
            print("SAVED MODEL", prefix)

    def save_cfg_to_json(self):
        return

    def save_cfg_to_yaml(self):
        return

    def setup_train_dataloader(self):
        # self.train_dataset.__len__ = lambda: self.num_steps
        sampler = None
        if self.is_dist_avail_and_initialized():
            sampler = DistributedSampler(self.train_dataset, shuffle=True, drop_last=self.drop_last)

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.device_train_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            sampler=sampler,
        )

    def setup_eval_dataloader(self):
        assert self.eval_dataset is not None
        sampler = None
        if self.is_dist_avail_and_initialized():
            sampler = DistributedSampler(
                self.eval_dataset,
                rank=self.local_rank,
                num_replicas=self.world_size,
                shuffle=True,
            )

        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.device_train_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            sampler=sampler,
        )

    def should_log_eval_metrics(self, step: int, max_step: int) -> bool:
        if step == 1:
            return True
        if step == max_step:
            return True
        return False

    def latest_metrics(self, pattern: str = ""):
        for k, m in self.metrics.items():
            if pattern not in k:
                continue
            if self.is_dist_avail_and_initialized() and "throughput" not in k:
                v = sync_and_compute(m)
            else:
                v = m.compute()
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.results[k].append((self.curr_step, v))

        return {k: v[-1][1] for k, v in self.results.items()}

    @torch.no_grad()
    def eval(self):
        print(f"{self.local_rank=}")
        if self.eval_dataloader is None or self.eval_dataset is None:
            raise ValueError("Eval dataset not set.")

        # reset metrics
        for k, m in self.metrics.items():
            if "eval" in k:
                m.reset()

        # zero gradients and set model to 'eval' mode.
        self.optimizer.zero_grad(set_to_none=True)
        self.model.eval()

        t0 = time.monotonic()
        for i, batch in enumerate(self.eval_dataloader):
            # prepare inputs
            batch = move_to_device(batch, self.device)

            # inference
            out = self.get_loss_and_metrics(self.model, batch)

            # update metrics
            for k, m in out.items():
                if f"eval/{k}" not in self.metrics:
                    self.metrics[f"eval/{k}"] = Mean(device=self.device)
                self.metrics[f"eval/{k}"].update(m)

            t1 = time.monotonic()
            self.metrics["eval/throughput"].update((i + 1) * self._device_eval_batch_size, t1 - t0)

        # log metrics
        print("---------------------->LOCAL RANK", self.local_rank, self.metrics)
        eval_metrics = self.latest_metrics("eval")
        print("finished computing eval metrics")
        return eval_metrics

    """
    Train
    """

    def should_train_log(self) -> bool:
        flag = False
        if self.curr_step % self.log_every == 0:
            flag = True
        return flag

    def should_accumulate_grad(self) -> bool:
        return self.curr_step % self.grad_accumulation_steps != 0

    def should_grad_step(self) -> bool:
        return self.curr_step % self.grad_accumulation_steps == 0

    def should_clip_grad(self) -> bool:
        return self.max_grad_norm is not None

    def should_checkpoint(self) -> bool:
        flag = False
        if self.curr_step % self.checkpoint_every == 0:
            flag = True
        if self.local_rank != 0:
            flag = False
        return flag

    def should_eval(self) -> bool:
        return self.curr_step % self.eval_every == 0

    def train(self):
        self.start_time = time.time()
        if self.profile:
            profiler = self.torch_profiler
        else:
            profiler = nullcontext()

        pbar = None
        if self.local_rank == 0:
            pbar = tqdm.tqdm(
                total=self.num_steps,
                desc="Training",
                position=0,
                leave=True,
            )

        if self.eval_on_start:
            eval_metrics = self.eval()
            self.log_metrics_to_console("eval", eval_metrics)

        for k, m in self.metrics.items():
            m.reset()

        t0 = time.monotonic()
        with profiler as p:
            for self.curr_step in range(self.start_step, self.num_steps):
                self.model.train()
                batch = next(iter(self.train_dataloader))
                batch = move_to_device(batch, self.device)
                out = self.get_loss_and_metrics(self.model, batch)

                # update metrics
                for k, m in out.items():
                    if f"train/{k}" not in self.metrics:
                        self.metrics[f"train/{k}"] = Mean(device=self.device)
                    self.metrics[f"train/{k}"].update(m)

                t1 = time.monotonic()
                self.metrics["train/throughput"].update((self.curr_step + 1) * self._device_eval_batch_size, t1 - t0)

                loss = out["loss"]
                loss.backward()

                if self.should_train_log():
                    latest_train_metrics = self.latest_metrics("train")
                    if self.local_rank == 0 and pbar is not None:
                        pbar.update(self.log_every)
                        pbar.set_postfix(**latest_train_metrics)

                    # TODO: append to file

                if self.should_clip_grad():
                    assert self.max_grad_norm is not None
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                if self.should_accumulate_grad():
                    continue

                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                if self.should_checkpoint():
                    self.save_checkpoint()

                print("RANK", self.local_rank, f"{self.metrics=}")
                if self.should_eval() and False:
                    eval_metrics = self.eval()
                    # BUG:stuck here
                    self.log_metrics_to_console("eval", eval_metrics)
                    if eval_metrics["eval/loss"] < self.best_eval_loss and self.checkpoint_best:
                        self.save_checkpoint("best")
                        self.save_model("best")

        if self.checkpoint_last:
            self.save_checkpoint("last")
            self.save_model("last")

        self.end_time = time.time()
        self.log.info(f"Training took {self.end_time - self.start_time:.1f} seconds.")

    """
    Dunder methods
    """

    def __repr__(self):
        inner_str = ""
        for k, v in self.__dict__.items():
            inner_str += f"\t{k}={v},\n"
        return f"""Trainer(\n{inner_str})"""
