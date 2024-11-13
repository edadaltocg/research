import logging

import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn, optim

from dnn.trainer_class import Trainer

log = logging.getLogger(__name__)


model = nn.Sequential(
    nn.Linear(768, 128), nn.ReLU(), nn.Linear(128, 2), nn.LogSoftmax(dim=-1)
)
X = torch.randn(1000, 768)
train_dataset = torch.utils.data.TensorDataset(X, torch.randint(0, 2, (1000,)))
eval_dataset = torch.utils.data.TensorDataset(X, torch.randint(0, 2, (1000,)))
optmizer = optim.SGD(model.parameters(), lr=0.01)


def get_labels(batch):
    return batch[1]


def get_inputs(batch):
    return batch[0]


def get_loss_and_metrics(model, batch):
    y = get_labels(batch)
    x = get_inputs(batch)
    # log.debug(f"{x.shape=}, {y.shape=}")
    # log.debug(f"{next(model.parameters()).device=}, {x.device=}, {y.device=}")
    pred = model(x)
    loss = F.cross_entropy(pred, y)
    return {"loss": loss, "acc": pred.argmax(-1) == y}


Trainer.seed_all(42)


def test_trainer_cpu_debug():
    t = Trainer(
        model,
        train_dataset,
        optmizer,
        get_loss_and_metrics,
        eval_dataset=eval_dataset,
        device_train_batch_size=32,
        device_eval_batch_size=100,
        debug_run=True,
        log_level=logging.DEBUG,
        compile=False,
        force_cpu=True,
    )
    m = t.eval()
    t.train()
    t.close()


def test_trainer_cpu():
    t = Trainer(
        model,
        train_dataset,
        optmizer,
        get_loss_and_metrics,
        eval_dataset=eval_dataset,
        num_steps=100,
        device_train_batch_size=32,
        device_eval_batch_size=100,
        compile=False,
        force_cpu=True,
        run_name="test_trainer_cpu",
        output_root="output/tests/train",
    )
    m = t.eval()
    t.train()
    t.close()


def test_trainer_cuda_ddp():
    t = Trainer(
        model,
        train_dataset,
        optmizer,
        get_loss_and_metrics,
        num_steps=100,
        eval_dataset=eval_dataset,
        device_train_batch_size=32,
        device_eval_batch_size=100,
        run_name="test_trainer_cuda",
        output_root="output/tests/train",
    )
    m = t.eval()
    t.train()
    t.close()


def test_trainer_fsdp():
    t = Trainer(
        model,
        train_dataset,
        optmizer,
        get_loss_and_metrics,
        num_steps=100,
        eval_dataset=eval_dataset,
        device_train_batch_size=32,
        device_eval_batch_size=100,
        checkpoint_best=False,
        checkpoint_last=False,
        num_checkpoints=0,
        compile=False,
        fsdp=True,
        run_name="test_trainer_fsdp",
        output_root="output/tests/train",
    )
    m = t.eval()
    t.train()
    t.close()


if __name__ == "__main__":
    import fire

    """
    torchrun --standalone --nnodes=1 --nproc_per_node=2 tests/test_trainer.py cuda
    """
    fire.Fire(
        {
            "cpu": test_trainer_cpu,
            "cuda": test_trainer_cuda_ddp,
            "ddp": test_trainer_cuda_ddp,
            "fsdp": test_trainer_fsdp,
        }
    )
