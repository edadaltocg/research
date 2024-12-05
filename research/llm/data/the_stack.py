import os
from pathlib import Path

from datasets import load_dataset


def get_dataset_online(limit="1%", split="train", dest="output/datasets/the_stack"):
    split = f"{split}[:{limit}]"
    dataset = load_dataset(
        "bigcode/the-stack-dedup",
        split=split,
        num_proc=os.cpu_count() // 2,
        trust_remote_code=True,
        # streaming=True,
    )
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(dest / split, num_proc=os.cpu_count() // 2)
    return dataset


if __name__ == "__main__":
    dataset = get_dataset_online()
    print(dataset)
