import matplotlib.pyplot as plt

from albumentations.pytorch import ToTensorV2
import copy

import albumentations as A
import numpy as np
import cv2


def dummy_image(w, h):
    return Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))


def visualize_from_path(image_path: str):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(6, 6))
    plt.axis("off")
    plt.imshow(image)


def visualize(image):
    plt.figure(figsize=(6, 6))
    plt.axis("off")
    plt.imshow(image)


def visualize_dataset(dataset, start_idx=0, samples=10, cols=5, show=False, dest=None):
    dataset = copy.deepcopy(dataset)
    dataset.transform = None
    rows = samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i in range(samples):
        image, _ = dataset[start_idx + i]
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    if dest:
        plt.savefig(dest)
    if show:
        plt.show()


def visualize_augmentations(dataset, idx=0, samples=10, cols=5, show=False, dest=None):
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    rows = samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i in range(samples):
        image, _ = dataset[idx]
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    if dest:
        plt.savefig(dest)
    if show:
        plt.show()
