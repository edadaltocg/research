import os
from pathlib import Path

from datasets import load_dataset

"""
C4
Dataset({
    features: ['text', 'timestamp', 'url'],
    num_rows: 3648689
})
"""


def get_dataset_online(limit="10%", split="train", dest="output/datasets/c4"):
    split = f"{split}[:{limit}]"
    dataset = load_dataset(
        "c4",
        "en",
        split=split,
        num_proc=os.cpu_count() - 1,
        trust_remote_code=True,
    )
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(dest / split):
        dataset.save_to_disk(dest / split)
    return dataset


if __name__ == "__main__":
    dataset = get_dataset_online()
    print(dataset)
