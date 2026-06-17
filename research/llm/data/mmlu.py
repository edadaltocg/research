import os
from pathlib import Path
from typing import Any

from datasets import load_dataset


def get_dataset_online(limit="1%", split="train", dest="output/datasets/mmlu"):
    split = f"{split}[:{limit}]"
    cpu_count = os.cpu_count()
    num_proc = cpu_count - 1 if cpu_count is not None else 1
    dataset: Any = load_dataset(
        "tasksource/mmlu",
        split=split,
        num_proc=num_proc,
        trust_remote_code=True,
    )
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(dest / split, num_proc=num_proc)
    return dataset


if __name__ == "__main__":
    dataset = get_dataset_online()
    print(dataset)
