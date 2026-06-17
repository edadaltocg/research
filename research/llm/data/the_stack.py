import os
from pathlib import Path
from typing import Any

from datasets import load_dataset


def get_dataset_online(limit="1%", split="train", dest="output/datasets/the_stack"):
    split = f"{split}[:{limit}]"
    cpu_count = os.cpu_count()
    num_proc = cpu_count // 2 if cpu_count is not None else 1
    dataset: Any = load_dataset(
        "bigcode/the-stack-dedup",
        split=split,
        num_proc=num_proc,
        trust_remote_code=True,
        # streaming=True,
    )
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(dest / split, num_proc=num_proc)
    return dataset


if __name__ == "__main__":
    dataset = get_dataset_online()
    print(dataset)
