import json
import math
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dnn.data
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.utils.data
import tqdm
import utils
from torch import Tensor  # tensor node in the computation graph
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullOptimStateDictConfig,
    FullStateDictConfig,
    MixedPrecision,
    StateDictType,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.utils.data.distributed import DistributedSampler
from utils import distrib_setup, get_date, get_rank, get_world_size, seed_all

fpSixteen = MixedPrecision(
    param_dtype=torch.float16,
    # Gradient communication precision.
    reduce_dtype=torch.float16,
    # Buffer precision.
    buffer_dtype=torch.float16,
)
fpSixteen_mixed = MixedPrecision(
    param_dtype=torch.float16,
    # Gradient communication precision.
    reduce_dtype=torch.float32,
    # Buffer precision.
    buffer_dtype=torch.float16,
)


bfSixteen = MixedPrecision(
    param_dtype=torch.bfloat16,
    # Gradient communication precision.
    reduce_dtype=torch.bfloat16,
    # Buffer precision.
    buffer_dtype=torch.bfloat16,
)

bfSixteen_mixed = MixedPrecision(
    param_dtype=torch.bfloat16,
    # Gradient communication precision.
    reduce_dtype=torch.float32,
    # Buffer precision.
    buffer_dtype=torch.bfloat16,
)


fp32_policy = MixedPrecision(
    param_dtype=torch.float32,
    # Gradient communication precision.
    reduce_dtype=torch.float32,
    # Buffer precision.
    buffer_dtype=torch.float32,
)


def get_lr(
    it: int, lr_decay_iters: int, learning_rate: float, warmup_iters: int, min_lr: float
):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def get_model_example():
    return nn.Linear(768, 2)


def get_train_dataset_example():
    return dnn.data.DummyDataset(n=1000, dim=768, num_classes=2)


def get_eval_dataset_example():
    return dnn.data.DummyDataset(n=100, dim=768, num_classes=2)


def get_optimizer_example(parameters):
    return optim.SGD(parameters, lr=0.01)


def get_lr_scheduler_example(optimizer, *args):
    return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


def get_loss_example(model, batch):
    # batch already in device
    x = batch["x"]
    y = batch["y"].squeeze()
    logits = model(x)
    return F.cross_entropy(logits, y, reduction="sum")


def standard_trainer(
    get_model: Callable[[], nn.Module],
    get_train_dataset: Callable[[], torch.utils.data.Dataset],
    get_eval_dataset: Callable[[], torch.utils.data.Dataset],
    get_optimizer: Callable[[Iterator[nn.Parameter]], optim.Optimizer],
    get_loss: Callable[[nn.Module, dict[str, Tensor]], Tensor],
    get_lr_scheduler: Callable[[optim.Optimizer, Any], Any] = None,
    gradient_accumulation_steps: int = 1,
    num_epochs: int = 3,
    num_workers: int = 2,
    batch_size: int = 10,
    model_prefix: str = "model",
    num_logs: int = 1000,
    num_evals: int = 10,
    num_regular_checkpoints: int = 5,
    dest_root: str = "output/train",
    seed: int = 42,
    max_steps: int | None = None,
    gradient_clip: bool = False,
):
    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = get_train_dataset()
    eval_dataset = get_eval_dataset()
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = get_model()
    model = model.to(device)
    print(model)
    print(model.parameters())

    optimizer = get_optimizer(model)
    # scheduler = get_lr_scheduler(optimizer) if get_lr_scheduler is not None else None
    scheduler = None

    total_num_steps = len(train_loader) * num_epochs
    print(f"Total number of steps: {total_num_steps}")
    if max_steps is None:
        max_steps = total_num_steps
    else:
        max_steps = min(max_steps, total_num_steps)
        total_num_steps = max_steps

    time_of_run = get_date()
    dest_path = Path(dest_root) / f"{model_prefix}/{time_of_run}"
    dest_path.mkdir(parents=True, exist_ok=True)

    pbar = tqdm.tqdm(
        total=total_num_steps, desc="Training step", position=0, leave=True
    )

    tracker = defaultdict(list)

    train_loss = torch.zeros(2, device=device)
    eval_loss = torch.zeros(2, device=device)
    best_eval_loss = float("inf")
    global_eval_loss = float("inf")
    global_train_loss = float("inf")

    eval_every = max(total_num_steps // num_evals, 1)
    log_every = max(total_num_steps // num_logs, 1)
    checkpoint_every = max(total_num_steps // num_regular_checkpoints, 1)
    print(
        f"Eval every: {eval_every}, Log every: {log_every}, Checkpoint every: {checkpoint_every}"
    )

    step = 0
    pbar.update(1)
    for epoch in range(1, num_epochs + 1):
        train_loss = torch.zeros(2, device=device)
        for batch in train_loader:
            model.train()
            k0 = list(batch.keys())[0]
            for key in batch.keys():
                batch[key] = batch[key].to(device)
            loss = get_loss(model, batch) / gradient_accumulation_steps
            loss.backward()
            if gradient_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if step % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
            train_loss[0] += loss.item()
            train_loss[1] += len(batch[k0])

            step += 1

            global_train_loss = train_loss[0].item() / train_loss[1].item()

            # Evaluating
            if step % eval_every == 0 or step == 1:
                model.eval()
                eval_loss = torch.zeros(2, device=device)
                for batch in eval_loader:
                    k0 = list(batch.keys())[0]
                    for key in batch.keys():
                        batch[key] = batch[key].to(device)
                    with torch.no_grad():
                        loss = get_loss(model, batch)
                    eval_loss[0] += loss.item()  # sum up batch loss
                    eval_loss[1] += len(batch[k0])

                global_eval_loss = eval_loss[0].item() / eval_loss[1].item()

            # Metrics (pre-allocate) TODO:
            tracker["mem_allocated"].append(
                utils.format_to_gb(torch.cuda.memory_allocated())
            )
            tracker["mem_reserved"].append(
                utils.format_to_gb(torch.cuda.memory_reserved())
            )
            tracker["train_loss"].append(global_train_loss)
            tracker["eval_loss"].append(global_eval_loss)
            tracker["step"].append(step)

            # Save best model
            if global_eval_loss < best_eval_loss:
                print(f"Saving best model at step {step}")
                save_name = f"{model_prefix}-best.pt"
                state = model.state_dict()
                torch.save(state, dest_path / save_name)
                best_cfg = {
                    "epoch": epoch,
                    "train_loss": global_train_loss,
                    "eval_loss": best_eval_loss,
                    "local_step": step,
                    "tracker": tracker,
                }
                torch.save(best_cfg, dest_path / f"{model_prefix}-best-cfg.pt")
                best_eval_loss = global_eval_loss

            # Checkpointing
            if step % checkpoint_every == 0 or step == 1 or step >= max_steps:
                state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "time_of_run": time_of_run,
                    "best_eval_loss": best_eval_loss,
                }

                print(f"Saving checkpoint at step {step}")
                save_name = f"{model_prefix}-{step}-ckpt.pt"
                torch.save(state, dest_path / save_name)

                if step >= max_steps:
                    break

            # Update progress bar and logging
            pbar.update(1)
            pbar.set_postfix(
                dict(
                    t_loss=tracker["train_loss"][-1],
                    e_loss=tracker["eval_loss"][-1] if "eval_loss" in tracker else "-",
                    mem=tracker["mem_allocated"][-1] + tracker["mem_reserved"][-1],
                )
            )

    pbar.close()

    results = {
        "num_epochs": num_epochs,
        "train_loss": global_train_loss,
        "eval_loss": global_eval_loss,
        "best_eval_loss": best_eval_loss,
        "local_step": step,
        "tracker": tracker,
    }
    with open(dest_path / f"{model_prefix}-results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("End of training")


def ddp_trainer():
    return

    # optimizer_state_dtype = ...
    # gradient_reduce_type = ...
    # peak_lr = ...
    # minimum_lr = ...
    #


def fsdp_trainer(
    get_model: Callable[[], nn.Module],
    get_train_dataset: Callable[[], torch.utils.data.Dataset],
    get_eval_dataset: Callable[[], torch.utils.data.Dataset],
    get_optimizer: Callable[[Iterator[nn.Parameter]], optim.Optimizer],
    get_lr_scheduler: Callable[[optim.Optimizer, Any], Any],
    get_loss: Callable[[nn.Module, dict[str, Tensor]], Tensor],
    gradient_accumulation_steps: int = 1,
    num_epochs: int = 3,
    num_workers: int = 2,
    local_batch_size: int = 10,
    model_prefix: str = "model",
    mixed_precision: bool = True,
    num_logs: int = 1000,
    num_evals: int = 10,
    num_regular_checkpoints: int = 5,
    dest_root: str = "output/train",
    seed: int = 42,
    max_steps: int | None = None,
    gradient_clip: bool | float = False,
    num_warmup_steps: int | None = None,
):
    """FSDP: Fully Sharded Data Parallelism.

    Shards model, optimizer state, and data across devices.

    Usage:
        Run from the cli as:
        ```bash
        torchrun --nnodes 1 --nproc_per_node 4
        ```

    References:
        [1] https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
        [2] https://pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html
    """
    # Distributed setup
    distrib_setup()
    seed_all(seed)
    rank = get_rank()
    world_size = get_world_size()

    # Data related setup
    train_dataset = get_train_dataset()
    eval_dataset = get_eval_dataset()
    train_sampler = DistributedSampler(
        train_dataset, rank=rank, num_replicas=world_size, shuffle=True
    )
    eval_sampler = DistributedSampler(eval_dataset, rank=rank, num_replicas=world_size)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=local_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        sampler=eval_sampler,
    )
    total_num_steps = len(train_loader) * num_epochs
    print(f"Total number of steps: {total_num_steps}")
    if max_steps is None:
        max_steps = total_num_steps
    else:
        max_steps = min(max_steps, total_num_steps)
        total_num_steps = max_steps

    # CUDA utils
    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    # Model related setup
    if utils.is_bf16_ready() and mixed_precision:
        mp_policy = bfSixteen
        precision = torch.bfloat16
    elif mixed_precision:
        mp_policy = fpSixteen
        precision = torch.float16
    else:
        mp_policy = fp32_policy  # defaults to fp32
        precision = torch.float32

    utils.print_rank0(f"Using mixed precision policy: {mp_policy}")
    model = get_model()
    model = FSDP(
        model,
        # cpu_offload=CPUOffload.AUTO,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # can increase the training speed in exchange for higher mem
        # sharding_strategy=ShardingStrategy.SHARD_GRAD_OP # ZERO2
        mixed_precision=mp_policy,
        device_id=torch.cuda.current_device(),
    )
    model_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    utils.print_rank0(model)

    # Optimizer related setup
    optimizer = get_optimizer(model.parameters())
    model = torch.compile(model)
    if num_warmup_steps is not None:
        scheduler = get_lr_scheduler(optimizer, max_steps, num_warmup_steps)
    else:
        scheduler = get_lr_scheduler(optimizer, max_steps)
    optimizer_save_policy = FullOptimStateDictConfig(
        offload_to_cpu=True, rank0_only=True
    )

    # Save results related setup
    time_of_run = get_date()
    dest_path = Path(dest_root) / f"{model_prefix}/{time_of_run}"
    if rank == 0:
        dest_path.mkdir(parents=True, exist_ok=True)

    # Training loop
    if rank == 0:
        pbar = tqdm.tqdm(total=total_num_steps, desc="Training")
        pbar.update(1)

    tracker = defaultdict(list)

    train_loss = torch.zeros(2, device=rank)
    eval_loss = torch.zeros(2, device=rank)
    best_eval_loss = float("inf")
    global_eval_loss = float("inf")
    global_train_loss = float("inf")

    eval_every = max(total_num_steps // num_evals, 1)
    log_every = max(total_num_steps // num_logs, 1)
    checkpoint_every = max(total_num_steps // num_regular_checkpoints, 1)
    print(
        f"Eval every: {eval_every}, Log every: {log_every}, Checkpoint every: {checkpoint_every}"
    )

    local_step = 0
    init_start_event.record(torch.cuda.current_stream())
    with torch.autocast("cuda", enabled=True, dtype=precision):
        for epoch in range(1, num_epochs + 1):
            train_sampler.set_epoch(epoch)
            train_loss = torch.zeros(2, device=rank)
            for batch in train_loader:
                model.train()
                k0 = list(batch.keys())[0]
                for key in batch.keys():
                    batch[key] = batch[key].to(rank)
                loss = get_loss(model, batch) / gradient_accumulation_steps
                loss.backward()
                if gradient_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                if local_step % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    if scheduler is not None:
                        scheduler.step()
                train_loss[0] += loss.item()
                train_loss[1] += len(batch[k0])

                local_step += 1

                # bottleneck?
                dist.reduce(train_loss, 0, op=dist.ReduceOp.SUM)
                global_train_loss = train_loss[0].item() / train_loss[1].item()

                # Evaluating
                if local_step % eval_every == 0 or local_step == 1:
                    model.eval()
                    eval_loss = torch.zeros(2, device=rank)
                    for batch in eval_loader:
                        k0 = list(batch.keys())[0]
                        for key in batch.keys():
                            batch[key] = batch[key].to(rank)
                        with torch.no_grad():
                            loss = get_loss(model, batch)
                        eval_loss[0] += loss.item()  # sum up batch loss
                        eval_loss[1] += len(batch[k0])

                    dist.reduce(eval_loss, 0, op=dist.ReduceOp.SUM)
                    global_eval_loss = eval_loss[0].item() / eval_loss[1].item()

                # Metrics
                if local_step % log_every == 0 or local_step == 1:
                    tracker["mem_allocated"].append(
                        utils.format_to_gb(torch.cuda.memory_allocated())
                    )
                    tracker["mem_reserved"].append(
                        utils.format_to_gb(torch.cuda.memory_reserved())
                    )
                    tracker["train_loss"].append(global_train_loss)
                    tracker["eval_loss"].append(global_eval_loss)
                    tracker["step"].append(local_step)

                # Save best model
                if global_eval_loss < best_eval_loss:
                    best_eval_loss = global_eval_loss
                    with FSDP.state_dict_type(
                        model, StateDictType.FULL_STATE_DICT, model_save_policy
                    ):
                        cpu_state = model.state_dict()
                    if rank == 0:
                        print(f"Saving best model at step {local_step}")
                        save_name = f"{model_prefix}-best.pt"
                        torch.save(cpu_state, dest_path / save_name)
                        best_cfg = {
                            "epoch": epoch,
                            "train_loss": global_train_loss,
                            "eval_loss": best_eval_loss,
                            "local_step": local_step,
                            "tracker": tracker,
                        }
                        torch.save(best_cfg, dest_path / f"{model_prefix}-best-cfg.pt")

                # Checkpointing
                if local_step % checkpoint_every == 0 or (local_step == 1 and False):
                    with FSDP.state_dict_type(
                        model,
                        StateDictType.FULL_STATE_DICT,
                        model_save_policy,
                        optimizer_save_policy,
                    ):
                        cpu_state = {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch,
                            "time_of_run": time_of_run,
                            "best_eval_loss": best_eval_loss,
                        }

                    if rank == 0:
                        print(f"Saving checkpoint at step {local_step}")
                        save_name = f"{model_prefix}-{local_step}-ckpt.pt"
                        torch.save(cpu_state, dest_path / save_name)

                    if local_step >= max_steps:
                        break

                # Update progress bar
                if rank == 0:
                    pbar.update(1)
                    pbar.set_postfix(
                        dict(
                            train_loss=tracker["train_loss"][-1],
                            eval_loss=tracker["eval_loss"][-1]
                            if "eval_loss" in tracker
                            else "-",
                        )
                    )

    init_end_event.record(torch.cuda.current_stream())

    if rank == 0:
        pbar.close()
        print(
            f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec"
        )
        print(f"{model}")

    dist.barrier()
    if rank == 0:
        # save results
        results = {
            "num_epochs": num_epochs,
            "train_loss": global_train_loss,
            "eval_loss": global_eval_loss,
            "best_eval_loss": best_eval_loss,
            "world_size": world_size,
            "local_step": local_step,
            "tracker": tracker,
        }
        with open(dest_path / f"{model_prefix}-results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("End of training")

    utils.distrib_cleanup()


@dataclass
class LRMonitor:
    optim: torch.optim.Optimizer

    def check(self) -> dict[str, float]:
        lrs = [group["lr"] for group in self.optim.param_groups]
        return {f"optim/learning_rate_group{idx}": lr for idx, lr in enumerate(lrs)}
