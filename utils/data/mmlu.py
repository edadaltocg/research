import os
from pathlib import Path

from datasets import load_dataset


def get_dataset_online(limit="1%", split="train", dest="output/datasets/mmlu"):
    split = f"{split}[:{limit}]"
    dataset = load_dataset(
        "tasksource/mmlu",
        split=split,
        num_proc=os.cpu_count() - 1,
        trust_remote_code=True,
    )
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(dest / split, num_proc=os.cpu_count() - 1)
    return dataset


if __name__ == "__main__":
    dataset = get_dataset_online()
    print(dataset)
