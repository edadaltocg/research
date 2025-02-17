from collections import defaultdict
import os
from typing import Any
import yaml
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.optim.lr_scheduler
import torch.utils.data
from torch.utils.data import DataLoader, Subset

from research.trainer.logger import TrainerLogger
from research.utils import get_criterion, get_optimizer, get_scheduler


@dataclass
class TrainerBase(ABC):
    model: nn.Module
    train_dataset: torch.utils.data.Dataset
    val_dataset: torch.utils.data.Dataset | None = None

    criterion: nn.Module | None = None
    criterion_cls: str | None = None
    criterion_kwargs: dict[str, Any] = field(default_factory=dict)

    optimizer: torch.optim.Optimizer | None = None
    optimizer_cls: str | None = None
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)

    lr_scheduler: torch.optim.lr_scheduler._LRScheduler | None = None
    lr_scheduler_cls: str | None = None
    lr_scheduler_kwargs: dict[str, Any] = field(default_factory=dict)

    num_epochs: int = 10
    gradient_clip_val: float | None = None
    accumulation_steps: int = 1
    enable_profiling: bool = False
    init_weights: str = "xavier_uniform"
    log_dir: str | Path = "logs"
    output_dir: str | Path = "output"
    log_interval: int = 10
    eval_interval: int = 1000
    save_interval: int = 5000
    compile_model: bool = False
    debug_run: bool = False
    overfit_run: bool = False

    checkpoint_prefix: str = ""
    checkpoint_dir: str | Path = "checkpoints"
    checkpoint_path: str | None = None
    num_checkpoints: int = 5
    resume_from_checkpoint: bool = False

    # data loader
    batch_size: int = 1
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = False
    prefetch_factor = None
    persistent_workers: bool = False

    def __post_init__(self):
        self.log_dir = Path(self.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.train_logger = TrainerLogger(self.log_dir)
        self._epoch = 0
        self._global_step = 0
        self.last_time = time.time()
        self.metrics = defaultdict(list)
        self.best_val_loss = float("inf")

        self._setup_device()
        self._setup_model()
        self._setup_optimizer()
        self._setup_lr_scheduler()
        self._setup_criterion()
        self._setup_dataloaders()
        self._init_training_state()
        if self.resume_from_checkpoint:
            self.load_checkpoint()

    @abstractmethod
    def _setup_device(self):
        pass

    @abstractmethod
    def _setup_model(self):
        pass

    def _setup_optimizer(self):
        if not self.optimizer and self.optimizer_cls:
            self.optimizer = get_optimizer(
                cls=self.optimizer_cls, parameters=self.model.parameters(), **self.optimizer_kwargs
            )

    def _setup_lr_scheduler(self):
        if not self.lr_scheduler and self.lr_scheduler_cls:
            self.lr_scheduler = get_scheduler(
                cls=self.lr_scheduler_cls, optimizer=self.optimizer, **self.lr_scheduler_kwargs
            )

    def _setup_criterion(self):
        if not self.criterion and self.criterion_cls:
            self.criterion = get_criterion(cls=self.criterion_cls, **self.criterion_kwargs)

    def _setup_dataloaders(self):
        if self.overfit_run:
            self.train_dataset = Subset(self.train_dataset, range(self.batch_size))

        self.train_loader = (
            DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=self.drop_last,
                prefetch_factor=self.prefetch_factor,
                persistent_workers=self.persistent_workers,
            )
            if self.train_dataset
            else None
        )

        self.val_loader = (
            DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                prefetch_factor=self.prefetch_factor,
                persistent_workers=self.persistent_workers,
            )
            if self.val_dataset
            else None
        )

    @abstractmethod
    def _init_training_state(self):
        pass

    @abstractmethod
    def _train_one_step(self, batch):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def _evaluate_one_step(self, batch):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    def tune(self):
        pass

    def profile(self):
        pass

    @abstractmethod
    def load_checkpoint(self):
        pass

    @abstractmethod
    def save_checkpoint(self):
        pass

    @abstractmethod
    def export_model(self):
        pass

    @classmethod
    def load_from_file(cls, config_file_path: str):
        with open(config_file_path) as file:
            config_dict = yaml.safe_load(file)

        optimizer_config = config_dict.pop("optimizer")
        scheduler_config = config_dict.pop("scheduler")
        criterion_config = config_dict.pop("criterion")
