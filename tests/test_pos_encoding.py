import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from dnn.modeling.pos_encoding import vanilla_positional_encoding
from utils import seed_all

seed_all(42)


def test_vanilla_positional_encoding():
    x = torch.Tensor(2, 50_000, 50)
    pe = vanilla_positional_encoding(x)
    assert pe.size() == (1, x.size(1), x.size(2))
    # plot
    fig = plt.figure(figsize=(4, 3), dpi=300)
    plt.imshow(pe[0].numpy(), cmap="viridis", aspect="auto")
    plt.title("Positional Encoding")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Sequence Position")
    plt.colorbar()
    plt.tight_layout()
    dest = Path(
        os.path.join("output", "tests", "plots", "vanilla_positional_encoding.png")
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(dest, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
