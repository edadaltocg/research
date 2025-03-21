import json
from collections import defaultdict
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
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
    train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)
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
    optimizer_save_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)

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
    print(f"Eval every: {eval_every}, Log every: {log_every}, Checkpoint every: {checkpoint_every}")

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
                    tracker["mem_allocated"].append(utils.format_to_gb(torch.cuda.memory_allocated()))
                    tracker["mem_reserved"].append(utils.format_to_gb(torch.cuda.memory_reserved()))
                    tracker["train_loss"].append(global_train_loss)
                    tracker["eval_loss"].append(global_eval_loss)
                    tracker["step"].append(local_step)

                # Save best model
                if global_eval_loss < best_eval_loss:
                    best_eval_loss = global_eval_loss
                    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, model_save_policy):
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
                            eval_loss=tracker["eval_loss"][-1] if "eval_loss" in tracker else "-",
                        )
                    )

    init_end_event.record(torch.cuda.current_stream())

    if rank == 0:
        pbar.close()
        print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
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
