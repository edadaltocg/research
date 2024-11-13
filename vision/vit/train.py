"""
The best results are obtained with supervised pre-training.
"""

import os

import torch
import torch.nn.functional as F
import torch.utils.data

import dnn
import dnn.trainer
from datasets import load_dataset, load_from_disk
from dnn.data import (
    MemoryMappedDataset,
    PreTransformImageClassificationDataset,
    make_image_pre_processor,
    make_image_processor,
)
from vit.model import VisionTransformer


def train_cifar10(
    name: str = "vit_base_16_224_cifar10",
    image_size=224,
    num_classes=10,
    patch_size=16,
    in_channels=3,
    num_layers=12,
    num_heads=12,
    hidden_dim=768,
    mlp_dim=3072,
    dropout_p=0.1,
    attn_dropout_p=0.1,
    num_epochs=300,
    batch_size=256,
    download=False,
    memmap=False,
):
    """Train a Vistion Transformer model on CIFAR10."""
    train_image_processor = make_image_processor(image_size, training=True)
    eval_image_processor = make_image_processor(image_size, training=False)
    pre_transform = make_image_pre_processor(256)

    # pre-processign steps
    # download cifar10 dataset
    if download:
        ds = load_dataset("cifar10", num_proc=os.cpu_count() - 1)  # type: ignore
        # save the dataset to disk
        # pre-process dataset
        ds.save_to_disk("output/datasets/cifar10")  # type: ignore

    # pre-process the dataset and save to disk as memory-mapped files
    if memmap:
        dataset = load_from_disk("output/datasets/cifar10")
        for split in ["train", "test"]:
            ds = PreTransformImageClassificationDataset(
                dataset[split], transform=pre_transform
            )
            dataloader = torch.utils.data.DataLoader(ds, batch_size=128, num_workers=2)
            MemoryMappedDataset.from_dataloader(
                dataloader, f"output/datasets/cifar10/{split}"
            )

    def _get_dataset(split="train", transform=train_image_processor):
        """
        Based on hf cifar10 dataset:
            {
              'img': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=32x32 at 0x201FA6EE748>,
              'label': 0
            }
        """
        memmap_dataset = MemoryMappedDataset(
            f"output/datasets/cifar10/{split}", transform=transform
        )
        return memmap_dataset

    def get_optimizer(model_parameters):
        return torch.optim.Adam(model_parameters, lr=1e-2, betas=(0.9, 0.999))

    def get_lr_scheduler(optimizer, num_steps: int, num_warmup_steps: int):
        linear_warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.5, total_iters=num_warmup_steps
        )
        cosine_decay = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_steps)
        schedulers = [linear_warmup, cosine_decay]
        return torch.optim.lr_scheduler.ChainedScheduler(schedulers)

    def get_loss(model, batch):
        x = batch["x"]
        y = batch["y"]
        outputs = model(x, y)
        return F.cross_entropy(outputs["logits"], y, reduction="sum")

    def get_model():
        model = VisionTransformer(
            num_classes=num_classes,
            patch_size=patch_size,
            in_channels=in_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout_p=dropout_p,
            attn_dropout_p=attn_dropout_p,
        )
        return model

    def get_train_dataset():
        return _get_dataset("train", transform=train_image_processor)

    def get_eval_dataset():
        return _get_dataset("test", transform=eval_image_processor)

    dnn.trainer.standard_trainer(
        get_model=get_model,
        get_train_dataset=get_train_dataset,
        get_eval_dataset=get_eval_dataset,
        get_optimizer=get_optimizer,
        # get_lr_scheduler=get_lr_scheduler,
        get_loss=get_loss,
        gradient_accumulation_steps=1,
        num_epochs=num_epochs,
        num_workers=2,
        batch_size=batch_size,
        model_prefix=name,
        # mixed_precision=True,
        num_logs=1000,
        num_evals=10,
        num_regular_checkpoints=5,
        dest_root="output/train",
        seed=42,
        max_steps=None,
    )


def train_imagenet():
    return
