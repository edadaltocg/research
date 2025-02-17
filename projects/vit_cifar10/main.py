from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torchvision
import yaml
from albumentations.pytorch import ToTensorV2
from torchinfo import summary

import research
from research.trainer.single_device import TrainerSingleDevice
from research.vision.vit.model import ViT
from research.vision.viz import visualize_dataset


def main(
    compile=False,
    profile=False,
    overfit_run=False,
    config_file_path: str = "config.yml",
    device="cpu",
    level="INFO",
    seed=42,
):
    research.utils.seed_all(seed)
    logs_path = Path(__file__).parent / research.config.LOGS_DIR / "main.out"
    log = research.logger.setup_logger(level, logs_path=logs_path)
    outs_dir = Path(__file__).parent / research.config.OUTPUT_DIR
    outs_dir.mkdir(exist_ok=True)

    # Begin

    image_size = 32
    train_transform = A.Compose([
        A.CenterCrop(height=image_size, width=image_size),
        # A.Rotate(limit=30, p=0.5),  # Randomly rotate images by up to 30 degrees
        # A.HorizontalFlip(p=0.5),  # Randomly flip images horizontally
        # A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        # A.RandomBrightnessContrast(p=0.5),
        # A.Blur(blur_limit=3, p=0.3),  # Apply blur to images
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    A.save(train_transform, outs_dir / "train_transform.json")
    # A.load('/tmp/transform.json')
    train_dataset = torchvision.datasets.CIFAR10(
        root=research.config.DATASETS_DIR,
        train=True,
        transform=lambda x: train_transform(image=np.array(x))["image"],
        download=True,
    )
    log.info("Train dataset ready")
    val_transform = A.Compose([
        A.CenterCrop(height=image_size, width=image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    A.save(val_transform, outs_dir / "val_transform.json")
    val_dataset = torchvision.datasets.CIFAR10(
        root=research.config.DATASETS_DIR,
        train=False,
        transform=lambda x: val_transform(image=np.array(x))["image"],
        download=True,
    )
    log.info("Val dataset ready")

    # plot images
    visualize_dataset(train_dataset, dest="output/train_ds_examples.png")

    with open(config_file_path) as file:
        config_dict = yaml.safe_load(file)

    model_config = config_dict.pop("model")
    with torch.device("meta"):
        model = ViT(num_classes=10, image_size=32, **model_config)
    log.info(summary(model))
    log.info("Model ready")

    optimizer_config = config_dict.pop("optimizer")
    scheduler_config = config_dict.pop("scheduler")
    criterion_config = config_dict.pop("criterion")

    trainer_config = config_dict.pop("trainer")
    trainer = TrainerSingleDevice(
        device=device,
        model=model,
        compile_model=compile,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        optimizer_cls=optimizer_config.pop("cls"),
        optimizer_kwargs=optimizer_config,
        lr_scheduler_cls=scheduler_config.pop("cls"),
        lr_scheduler_kwargs=scheduler_config,
        criterion_cls=criterion_config.pop("cls"),
        criterion_kwargs=criterion_config,
        overfit_run=overfit_run,
        **trainer_config,
    )
    x = torch.randn(1, 3, 32, 32, device=device)
    pred = model(x)
    log.debug(f"{pred=}")
    if profile:
        trainer.profile()
    else:
        log.info("Train Vision Transformers on CIFAR-10")
        trainer.train()

    log.info("Done!")


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
