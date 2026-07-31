"""Train Sudoku solver on the dataset"""

from collections import OrderedDict
from dataclasses import dataclass

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ebm_sudoku_solver.solver.ebm.dataset import SudokuDataset
from ebm_sudoku_solver.solver.ebm.model import EBMSudokuVerifierModel
from ebm_sudoku_solver.solver.ebm.utils import get_logger
from research.utils import seed_all


def _replay_buffer():
    # TODO:
    return


def _save_checkpoint():
    return {"model", "optimizer", "lr_scheduler"}


@torch.no_grad()
def _update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def evaluation():
    return


@dataclass(kw_only=True)
class Hyperparameters:
    lr: float
    warmup_steps: int = 5
    weight_decay: float


@dataclass(kw_only=True)
class TrainWorkerConfig:
    num_epochs: int
    max_steps: int
    batch_size: int
    dataset_length: int

    model_embed_dim: int
    model_hidden_dim: int

    num_workers: int

    hparams: Hyperparameters


def train_worker_vanilla(config: TrainWorkerConfig):
    dataset = SudokuDataset(config.dataset_length)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    model = EBMSudokuVerifierModel(embed_dim=config.model_embed_dim, hidden_dim=config.model_hidden_dim)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.hparams.lr)

    total_steps = len(dataset) // config.batch_size * config.num_epochs

    lr_scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.01, total_iters=config.hparams.warmup_steps),
            CosineAnnealingLR(optimizer, T_max=total_steps - config.hparams.warmup_steps, eta_min=1e-6),
        ],
        milestones=[config.hparams.warmup_steps],
    )

    model.train()
    epochs_pbar = tqdm(range(config.num_epochs))
    for _ in epochs_pbar:
        # Train epoch.
        for batch in dataloader:
            output = model(batch["input"])
            loss = loss_fn(output, batch["label"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

    return


@dataclass
class TrainConfig:
    worker: TrainWorkerConfig
    seed: int = 42


def train(config: TrainConfig):
    logger = get_logger()
    seed_all()
    # TODO: load from checkpoint
    train_worker_vanilla(config.worker)
    # TODO: instantiale callbacks
    _save_checkpoint()


if __name__ == "__main__":
    logger = get_logger()
    logger.debug("hi")
    # train(TrainConfig(worker=TrainWorkerConfig()))
