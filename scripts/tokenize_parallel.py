import os
import concurrent.futures
from functools import partial
from tqdm import tqdm
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer

from datasetsutils.common import get_text_pretrain_dataset


def get_dataset_slice(dataset, start, end):
    for i in range(start, end):
        yield dataset[i]


tokenizer = AutoTokenizer.from_pretrained("output/weights/llama2-7b")
dataset = get_text_pretrain_dataset()

dest = "output/datasets/mypile_tokenized"
dest = Path(dest)
num_files = os.cpu_count() - 1
indexes = range(0, len(dataset), len(dataset) // num_files)
partial_datasets = [
    get_dataset_slice(dataset, start, end)
    for start, end in zip(indexes[:-1], indexes[1:])
]
partial_total = len(dataset) // num_files


def write_tokenized_dataset_in_parallel(index):
    tokenizer_fn = partial(tokenizer, padding=False, truncation=False)
    partial_dataset = partial_datasets[index]
    generator = map(tokenizer_fn, partial_dataset)
    with open(dest / f"tokens_{index}.bin", "wb") as f:
        for tokens in tqdm(
            generator, total=partial_total, desc=f"{index}", position=index
        ):
            tokens = np.array(tokens.input_ids, dtype=np.uint16)
            f.write(tokens.tobytes())
            f.flush()


def main():
    with concurrent.futures.ProcessPoolExecutor(os.cpu_count() - 1) as executor:
        for _ in executor.map(write_tokenized_dataset_in_parallel, range(num_files)):
            pass


if __name__ == "__main__":
    main()