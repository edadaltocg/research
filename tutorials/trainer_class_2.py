import logging
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Literal, Optional, TypeVar

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.utils.data
from numpy._typing import _UnknownType
from torch.distributed.fsdp import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import ProfilerActivity, schedule
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from datasetsutils.common import build_pre_train_dataset

from llama2.model import CalmLlama2HF, llama_1b_config, prepare_calm_llama

# from llama2.model import prepare_calm_llama

T = TypeVar("T")
log = logging.getLogger(__name__)


def get_criterion_cls(criterion_name: str) -> nn.modules.loss._Loss:
    return getattr(nn, criterion_name)


def get_optimizer_cls(optimizer_name: str) -> optim.Optimizer:
    return getattr(optim, optimizer_name)


def get_scheduler_cls(scheduler_name: str) -> optim.lr_scheduler._LRScheduler:
    return getattr(optim.lr_scheduler, scheduler_name)


def move_to_device(o: T, device: torch.device) -> T:
    if isinstance(o, torch.Tensor):
        return o.to(device)  # type: ignore[return-value]
    elif isinstance(o, dict):
        return {k: move_to_device(v, device) for k, v in o.items()}  # type: ignore[return-value]
    elif isinstance(o, list):
        return [move_to_device(x, device) for x in o]  # type: ignore[return-value]
    elif isinstance(o, tuple):
        return tuple(move_to_device(x, device) for x in o)  # type: ignore[return-value]
    else:
        return o


@dataclass
class Trainer:
    model: nn.Module
    train_forward: Callable
    train_dataset: Dataset | IterableDataset
    eval_dataset: Optional[Dataset | IterableDataset] = None
    eval_forward: Optional[Callable] = None
    grad_norm: Optional[float] = 1.0

    fsdp: bool = True
    auto_wrap_policy: Optional[_UnknownType] = None
    optimizer_name: str = "AdamW"
    peak_lr: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-5

    lr_scheduler_name: str = "linear"
    warmup_steps: int = 2000
    minimum_lr: float = 3e-5

    num_steps: int = 10000
    start_step: int = 1

    desired_batch_size: int = 32
    device_train_batch_size: int = 32
    device_eval_batch_size: int = 32

    distrib_backend: str = "nccl"
    master_addr: str = "localhost"
    master_port: str = "12355"
    distrib_init_method: str = "env://"

    num_workers: int = 8
    drop_last: bool = True

    param_dtype: Literal["float16", "bfloat16", "float32"] = "float32"
    ### gradient communication precision.
    reduce_dtype: Literal["float16", "bfloat16", "float32"] = "float32"
    ### buffer precision.
    buffer_dtype: Literal["float16", "bfloat16", "float32"] = "float32"

    compile: bool = True
    compiler_mode: Literal[
        "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"
    ] = "default"
    compiler_fullgraph: bool = False
    compiler_backend: str = "inductor"

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

    profile: bool = False
    force_cpu: bool = False
    run_name: str = ""
    output_root: str = "output/train"
    seed: int = 42

    def __post_init__(self):
        torch.set_float32_matmul_precision("high")
        torch.set_default_dtype(self.dtype_converter(self.param_dtype))
        dest_root = "output/tutorials/locals"
        dest_root = Path(dest_root)
        dest_root.mkdir(parents=True, exist_ok=True)
        self.dest_root = dest_root
        self.setup_distrib()
        seed = self.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        logging.basicConfig(
            level=logging.INFO if self.RANK == 0 else logging.ERROR,
            format=self.log_format,
        )

        self.pbar = self.setup_pbar()
        self.train_dataloader = self.setup_train_dataloader()
        self.train_iter = iter(self.train_dataloader)
        self.eval_dataloader = self.setup_eval_dataloader()
        self.model = self.setup_model()
        # log.info(f"{self.model.dtype=}")
        if self.compile:
            self.model = self.compile_model()  # type: ignore
        self.optimizer = self.setup_optimizer()
        self.scheduler = self.setup_lr_scheduler()
        self.train_history = {}
        self.eval_history = {}
        self.train_latest_history = {}
        self.eval_latest_history = {}
        self.writer = SummaryWriter(log_dir=dest_root / "train-logs", comment="train")
        self.profiler = self.setup_profiler()

        self.num_logs = max(1, self.num_logs)
        self.log_freq = max(self.num_steps // self.num_logs, 1)
        if self.num_checkpoints >= 1:
            self.checkpoint_freq = max(self.num_steps // self.num_checkpoints, 1)
        else:
            self.checkpoint_freq = math.inf
        self.eval_freq = max(self.num_steps // self.num_evals, 1)
        self.gradient_accumulation_steps = max(
            1, self.desired_batch_size // self.device_train_batch_size
        )

    def setup_pbar(self):
        return tqdm(
            range(self.start_step, self.num_steps + 1),
            dynamic_ncols=True,
            position=self.RANK,
            desc=f"Train [{self.RANK+1}/{self.WORLD_SIZE}]",
            leave=True,
        )

    def rng_state(self):
        return {
            "np": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "random": random.getstate(),
        }

    def setup_distrib(self):
        log.info("Setting up distributed training")
        try:
            dist.init_process_group(
                backend=self.distrib_backend, init_method=self.distrib_init_method
            )
            self.RANK = dist.get_rank()
            self.WORLD_SIZE = dist.get_world_size()
            self.device = torch.device(f"cuda:{self.RANK}")
        except ValueError:
            self.RANK = 0
            self.WORLD_SIZE = 1 if torch.cuda.is_available() else 0
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.force_cpu:
            self.WORLD_SIZE = 1
            self.RANK = 0
            self.device = torch.device("cpu")
        log.info(f"Using {self.device=}, {self.RANK=}, {self.WORLD_SIZE=}")

    def setup_train_dataloader(self):
        log.info(f"Setting up train dataloader for {self.device=}")
        if self.WORLD_SIZE > 1:
            sampler = DistributedSampler(
                self.train_dataset,
                drop_last=self.drop_last,
                rank=self.RANK,
                num_replicas=self.WORLD_SIZE,
            )
        else:
            sampler = None
        log.info(f"{sampler=}")
        return DataLoader(
            self.train_dataset,
            batch_size=self.device_train_batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=self.drop_last,
            # generator=torch.Generator(self.device.type),
        )

    def setup_eval_dataloader(self):
        log.info(f"Setting up eval dataloader for {self.device=}")
        if self.eval_dataset is None or self.eval_forward is None:
            return None
        if self.WORLD_SIZE > 1:
            sampler = DistributedSampler(
                self.eval_dataset,
                drop_last=False,
                rank=self.RANK,
                num_replicas=self.WORLD_SIZE,
            )
        else:
            sampler = None
        log.info(f"{sampler=}")
        return DataLoader(
            self.eval_dataset,
            batch_size=self.device_eval_batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            # generator=torch.Generator(self.device.type),
        )

    def dtype_converter(self, dtype):
        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]

    @property
    def precision(self):
        return MixedPrecision(
            param_dtype=self.dtype_converter(self.param_dtype),
            reduce_dtype=self.dtype_converter(self.reduce_dtype),
            buffer_dtype=self.dtype_converter(self.buffer_dtype),
        )

    def setup_model(self):
        log.info("Setting up model")
        non_trainable_params = sum(
            p.numel() for p in self.model.parameters() if not p.requires_grad
        )
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        log.info(f"Num trainable params: {trainable_params:,}")
        log.info(f"Num non trainable params: {non_trainable_params:,}")
        if self.WORLD_SIZE > 1:
            if self.fsdp:
                log.info("Setting up FSDP")
                self.model = FSDP(
                    self.model,
                    device_id=self.RANK,
                    # backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                    # sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
                    sharding_strategy=ShardingStrategy.FULL_SHARD,
                    mixed_precision=self.precision,
                    use_orig_params=True,
                    limit_all_gathers=True,
                    forward_prefetch=False,
                    auto_wrap_policy=self.auto_wrap_policy,
                    # cpu_offload=CPUOffload(True),
                )
            else:
                log.info(f"Setting up DDP on {self.device=}")
                self.model = self.model.to(self.device)
                self.model = DDP(self.model, device_ids=[self.RANK])
        else:
            log.info(f"Setting up model on single device, {self.device=}")
            self.model = self.model.to(
                self.device, dtype=self.dtype_converter(self.param_dtype)
            )

        return self.model

    def setup_optimizer(self):
        log.info("Setting up optimizer")
        return optim.AdamW(
            self.model.parameters(),
            lr=self.peak_lr,
            weight_decay=self.weight_decay,
            betas=(self.beta1, self.beta2),
            eps=self.epsilon,
        )

    def setup_lr_scheduler(self):
        log.info("Setting up lr scheduler")
        linear_warmup = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=self.minimum_lr / self.peak_lr,
            total_iters=self.warmup_steps,
        )

        cosine_decay = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, self.num_steps
        )
        schedulers = [linear_warmup, cosine_decay]
        return optim.lr_scheduler.ChainedScheduler(schedulers)

    def compile_model(self):
        log.info("Compiling model")
        return torch.compile(
            self.model,
            mode=self.compiler_mode,
            backend=self.compiler_backend,
            fullgraph=self.compiler_fullgraph,
        )

    # def resume_from_checkpoint(self, checkpoint_path: Optional[str] = None):
    #     if checkpoint_path:
    #         self.checkpoint_path = checkpoint_path
    #         checkpoint = self.load_checkpoint(checkpoint_path)
    #         self.state_dict = checkpoint

    def every(self, freq, func, *args, **kwargs):
        if self.step % freq == 0:
            return func(*args, **kwargs)

    def clip_grad(self):
        if self.grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)

    def log_train(self):
        for k, v in self.train_outputs.items():
            if k not in self.train_history:
                self.train_history[k] = torch.zeros(
                    (self.num_steps + 1,), device=self.device
                )
            self.train_history[k][self.step] = v
        if self.RANK <= 1:
            self.train_latest_history = {}
            for k, v in self.train_history.items():
                vv = v[self.step].item()
                self.train_latest_history[f"train/{k}"] = vv
                self.writer.add_scalar(f"train/{self.RANK}/{k}", vv, self.step)
            self.pbar.set_postfix(
                **self.train_latest_history, **self.eval_latest_history
            )

    def log_eval(self):
        if self.eval_dataset is None or self.eval_forward is None:
            return
        for k, v in self.eval_outputs.items():
            if k not in self.eval_history:
                self.eval_history[k] = torch.zeros(
                    (self.num_steps + 1,), device=self.device
                )
            self.eval_history[k][self.step] = v
        if self.RANK <= 1:
            self.eval_latest_history = {}
            for k, v in self.eval_history.items():
                vv = v[self.step].item()
                self.eval_latest_history[f"eval/{k}"] = vv
                self.writer.add_scalar(f"eval/{self.RANK}/{k}", vv, self.step)

    def log_optimizer(self):
        return

    def gradient_step(self):
        log.info(f"Gradient step at {self.step=}")
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def forward_backward(self):
        self.model.train()
        try:
            self.train_batch = next(self.train_iter)
        except StopIteration:
            # StopIteration is thrown if dataset ends
            # reinitialize data loader
            self.train_iter = iter(self.train_dataloader)
            self.train_batch = next(self.train_iter)

        self.train_batch = move_to_device(self.train_batch, self.device)
        self.train_outputs = self.train_forward(self.model, self.train_batch)
        loss = self.train_outputs["loss"] / self.gradient_accumulation_steps
        loss.backward()

    @torch.no_grad()
    def eval(self):
        if self.eval_forward is None or self.eval_dataset is None:
            return
        self.model.eval()
        for eval_batch in self.eval_dataloader:
            self.eval_batch = move_to_device(eval_batch, self.device)
            self.eval_outputs = self.eval_forward(self.model, self.eval_batch)

    def checkpoint(self):
        if self.WORLD_SIZE > 1:
            with FSDP.state_dict_type(
                self.model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
                FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
            ):
                model_state_dict = self.model.state_dict()
                optimizer_state_dict = self.optimizer.state_dict()
                scheduler_state_dict = self.scheduler.state_dict()
            for _, v in self.train_history.items():
                dist.reduce(v, 0, op=dist.ReduceOp.AVG)
            for _, v in self.eval_history.items():
                dist.reduce(v, 0, op=dist.ReduceOp.AVG)
            dist.barrier()
        else:
            model_state_dict = self.model.state_dict()
            optimizer_state_dict = self.optimizer.state_dict()
            scheduler_state_dict = self.scheduler.state_dict()

        self.checkpoint_metrics = {
            **{k: v[self.step].item() for k, v in self.train_history.items()},
            **{k: v[self.step].item() for k, v in self.eval_history.items()},
        }
        checkpoint = {
            "model": model_state_dict,
            "optimizer": optimizer_state_dict,
            "scheduler": scheduler_state_dict,
            "train_history": {k: v[: self.step] for k, v in self.train_history.items()},
            "eval_history": {k: v[: self.step] for k, v in self.eval_history.items()},
            "checkpoint_metrics": self.checkpoint_metrics,
            **self.trainer_hparams,
        }
        if self.RANK == 0:
            path = self.dest_root / f"{self.checkpoint_prefix}-{self.step}.pt"
            log.info(f"Checkpointing at {self.step=} to {path=}")
            torch.save(checkpoint, str(path))
            self.writer.add_hparams(self.trainer_hparams, self.checkpoint_metrics)

    def run(self):
        if self.eval_on_start:
            self.eval()
        with self.profiler as _:
            for self.step in self.pbar:
                self.every(1, self.forward_backward)
                self.every(1, self.clip_grad)
                self.every(self.log_freq, self.log_train)
                self.every(self.gradient_accumulation_steps, self.gradient_step)
                self.every(self.gradient_accumulation_steps, self.log_optimizer)
                self.every(self.eval_freq, self.eval)
                self.every(self.eval_freq, self.log_eval)
                self.every(self.checkpoint_freq, self.checkpoint)
        if self.eval_on_end:
            self.eval()
        if self.checkpoint_last:
            self.checkpoint()
        self.writer.close()
        if self.WORLD_SIZE > 1:
            dist.barrier()
            dist.destroy_process_group()

    @property
    def trainer_hparams(self):
        d = {}
        for k, v in self.__dict__.items():
            if type(v) in [int, float, str, bool]:
                d[k] = v
        return d

    def state_dict(self):
        d = {}
        for k, v in self.__dict__.items():
            if type(v) in [int, float, str, bool]:
                d[k] = v
            if k == "model":
                d[k] = v.state_dict()
            if k == "optimizer":
                d[k] = v.state_dict()
            if k == "scheduler" and v:
                d[k] = v.state_dict()

        return d

    def load_state_dict(self, d):
        for k, v in d.items():
            setattr(self, k, v)

    def setup_profiler(self):
        if self.profile:
            profiling_schedule = schedule(wait=1, warmup=5, active=3, repeat=1)
            on_trace_ready = torch.profiler.tensorboard_trace_handler(
                str(self.dest_root / "trace")
            )
            activities = [ProfilerActivity.CPU]
            if self.WORLD_SIZE >= 1:
                activities.append(ProfilerActivity.CUDA)
            torch_profiler = torch.profiler.profile(
                activities=activities,
                record_shapes=False,
                profile_memory=False,
                with_stack=True,
                schedule=profiling_schedule,
                on_trace_ready=on_trace_ready,
            )
            return torch_profiler
        return nullcontext()

    @staticmethod
    def train_forward_example(model, batch):
        x = batch[0][:, :-1].contiguous()
        y = batch[0][:, 1:].clone().contiguous()
        outputs = model.forward(x)
        logits = outputs.logits
        logits = logits.view(-1, logits.size(-1))
        y = y.view(-1)
        loss = F.cross_entropy(logits, y)
        return {"loss": loss}

    @staticmethod
    def eval_forward_example(model, batch):
        x = batch[0][:, :-1]
        y = batch[0][:, 1:].clone().to(torch.int64)
        t0 = time.time()
        outputs = model.forward(x)
        t1 = time.time()
        logits = outputs.logits
        logits = logits.view(-1, logits.size(-1))
        y = y.view(-1)
        loss = F.cross_entropy(logits, y)
        ppl = torch.exp(loss)
        throughput = torch.tensor(x.size(0) / (t1 - t0), device=x.device)
        return {"loss": loss, "throughput": throughput, "ppl": ppl}

    def __repr__(self):
        inner_str = ""
        for k, v in self.__dict__.items():
            inner_str += f"\t{k}={v},\n"
        return f"""Trainer(\n{inner_str})"""


def test_trainer(
    num_calm_layers=32,
    fsdp=False,
    seq_len=2048,
    vocab_size=100,
    num_samples=1000,
    num_steps=1000,
    batch_size=5,
    d_model=768,
    nhead=8,
    num_layers=8,
    dim_feedforward=3072,
    bias=False,
    grad_clip_norm=1,
    desired_batch_size=1024,
    num_workers=8,
    compile=False,
    profile=False,
    force_cpu=False,
    flag=0,
):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def train_forward_example(model, batch):
        x = batch[0]  # [:, :-1].contiguous()
        assert x.device.type == "cuda", f"{x.device=}"
        assert model.device.type == "cuda", f"{model.device=}"
        outputs, extra_logits = model.forward(x, labels=True)
        loss = outputs.loss
        return {"loss": loss}

    def eval_forward_example(model, batch):
        x = batch[0]  # [:, :-1].contiguous()
        assert x.device.type == "cuda", f"{x.device=}"
        assert model.device.type == "cuda", f"{model.device=}"
        outputs, extra_logits = model.forward(x, labels=True)
        loss = outputs.loss
        ppl = torch.exp(loss)
        return {"loss": loss, "ppl": ppl}

    cfg = llama_1b_config
    cfg.vocab_size = vocab_size + 1
    tokenizer = None
    auto_wrap_policy = None
    if flag == 1:
        model = LlamaForCausalLM(cfg)
        train_forward = Trainer.train_forward_example
        eval_forward = Trainer.eval_forward_example
    elif flag == 2:
        model = CalmLlama2HF(cfg, 4)
        train_forward = train_forward_example
        eval_forward = eval_forward_example
    else:
        model, tokenizer = prepare_calm_llama(num_layers=num_calm_layers)
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                LlamaDecoderLayer,
            },
        )
        train_forward = train_forward_example
        eval_forward = eval_forward_example

    dataset = []
    for i in range(num_samples):
        assert vocab_size < seq_len
        elems = [torch.arange(0, vocab_size)] * (2 * seq_len // vocab_size)
        elem = torch.cat(elems)[: seq_len + 1]
        assert elem.shape == (seq_len + 1,)
        assert elem.device == torch.device("cpu")
        if tokenizer is not None:
            elem = " ".join([str(e.item()) for e in elem])
            elem = tokenizer.encode(elem, return_tensors="pt")
            elem = elem[:, : seq_len + 1]
            assert elem.shape == (1, seq_len + 1)
            dataset.append(elem)
        else:
            dataset.append(elem.unsqueeze(0))
    dataset = torch.utils.data.TensorDataset(torch.concatenate(dataset))
    train_dataset = torch.utils.data.Subset(dataset, range(0, 80))
    eval_dataset = torch.utils.data.Subset(dataset, range(80, 100))
    print(train_dataset[:5])
    print(f"{train_dataset[0][0].shape}")

    my_trainer = Trainer(
        model=model,
        fsdp=fsdp,
        auto_wrap_policy=auto_wrap_policy,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        train_forward=train_forward,
        eval_forward=eval_forward,
        device_train_batch_size=batch_size,
        device_eval_batch_size=batch_size,
        num_steps=num_steps,
        desired_batch_size=batch_size,
        num_evals=10,
        grad_norm=grad_clip_norm,
        num_workers=num_workers,
        compile=compile,
        profile=profile,
        num_checkpoints=0,
        num_logs=num_samples,
        param_dtype="bfloat16",
        buffer_dtype="bfloat16",
        force_cpu=force_cpu,
    )
    my_trainer.run()

    elem = torch.arange(0, vocab_size, device=my_trainer.device).unsqueeze(0)
    if flag == 1:
        y = my_trainer.model(elem)
        logits_final = y.logits
        extra_preds = []
    elif flag == 2:
        y = my_trainer.model(elem, labels=True)
        logits_final = y[0].logits
        extra_logits = y[1]
        extra_preds = [logits.argmax(-1).cpu() for logits in extra_logits]
    else:
        elem = "1 2 3 4 5 6"
        input_ids = tokenizer.encode(elem, return_tensors="pt")
        y = my_trainer.model(input_ids, labels=True)
        logits_final = y[0].logits
        extra_logits = y[1]
        extra_preds = [
            logits.argmax(-1).cpu().numpy().tolist() for logits in extra_logits
        ]
    print(f"{elem=}")
    pred_final = logits_final.argmax(-1).cpu().numpy().tolist()
    print(f"{pred_final=}")
    print(f"{extra_preds=}")


def calm_trainer(
    num_calm_layers=32,
    num_steps=1000,
    grad_clip_norm=1,
    batch_size=1024,
    num_workers=8,
    compile=True,
    profile=False,
):
    def train_forward_example(model, batch):
        x = batch[0]  # [:, :-1].contiguous()
        assert x.device.type == "cuda", f"{x.device=}"
        assert model.device.type == "cuda", f"{model.device=}"
        outputs, extra_logits = model.forward(x, labels=True)
        loss = outputs.loss
        return {"loss": loss}

    def eval_forward_example(model, batch):
        x = batch[0]  # [:, :-1].contiguous()
        assert x.device.type == "cuda", f"{x.device=}"
        assert model.device.type == "cuda", f"{model.device=}"
        outputs, extra_logits = model.forward(x, labels=True)
        loss = outputs.loss
        ppl = torch.exp(loss)
        return {"loss": loss, "ppl": ppl}

    model, tokenizer = prepare_calm_llama(num_layers=num_calm_layers)
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            LlamaDecoderLayer,
        },
    )
    train_forward = train_forward_example
    eval_forward = eval_forward_example
    dataset = build_pre_train_dataset(
        {
            "c4": 1,
        }
    )
    train_dataset = torch.utils.data.Subset(dataset, range(0, 80))
    eval_dataset = torch.utils.data.Subset(dataset, range(80, 100))
    my_trainer = Trainer(
        model=model,
        fsdp=True,
        auto_wrap_policy=auto_wrap_policy,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        train_forward=train_forward,
        eval_forward=eval_forward,
        device_train_batch_size=batch_size,
        device_eval_batch_size=batch_size,
        num_steps=num_steps,
        desired_batch_size=batch_size,
        num_evals=10,
        grad_norm=grad_clip_norm,
        num_workers=num_workers,
        compile=compile,
        profile=profile,
        num_checkpoints=0,
        num_logs=num_steps,
        param_dtype="bfloat16",
        buffer_dtype="bfloat16",
        force_cpu=False,
    )
    my_trainer.run()


if __name__ == "__main__":
    """
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --standalone --nnodes=1 --nproc_per_node=2

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --standalone --nnodes=1 --nproc_per_node=2 tutorials/trainer_class_2.py calm
    """
    import fire

    fire.Fire(
        {
            "test": test_trainer,
            "calm": calm_trainer,
        }
    )
