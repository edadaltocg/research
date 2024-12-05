import os
from pathlib import Path

from datasets import load_dataset

"""
Dataset({
    features: ['id', 'url', 'title', 'text'],
    num_rows: 64587
})
"""


def get_dataset_online(limit="1%", split="train", dest="output/datasets/wikipedia"):
    split = f"{split}[:{limit}]"
    dataset = load_dataset(
        "wikipedia",
        "20220301.en",
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
    dataset = get_dataset_online(limit="100%")
    print(dataset)
