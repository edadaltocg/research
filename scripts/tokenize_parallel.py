import os
import concurrent.futures
from functools import partial
from tqdm import tqdm
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer

from datasetsutils.common import get_text_pretrain_partial_dataset


def get_dataset_slice(dataset, start, end):
    for i in range(start, end):
        yield dataset[i]


def write_tokenized_dataset_in_parallel(index):
    # global tokenizer, partial_datasets, partial_total, eos_id, dest, num_files
    tokenizer_fn = partial(
        tokenizer, padding=False, truncation=False, add_special_tokens=True
    )
    partial_dataset = partial_datasets[index]
    generator = map(tokenizer_fn, partial_dataset)
    chunk = []
    i = 0
    with open(dest / f"tokens_{index + 1:03d}-{num_files:03d}.bin", "wb") as f:
        for tokens in tqdm(
            generator,
            total=partial_total,
            desc=f"{index}",
            position=index,
            dynamic_ncols=True,
        ):
            chunk += tokens.input_ids + [eos_id]
            if i == 0:
                print(f"chunk={chunk[:10] + ['...'] + chunk[-10:]}")
                i += 1
            if len(chunk) > 1024 * 1024:
                tokens = np.array(chunk, dtype=np.uint16)
                f.write(tokens.tobytes())
                f.flush()
                chunk = []


if __name__ == "__main__":
    # went from 1h10min to 10min
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_files", type=int, default=os.cpu_count() // 2)
    parser.add_argument("--root", type=str, default="output/datasets")
    parser.add_argument("--dataset", type=str, default="c4")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--limit", type=str, default="100%")
    parser.add_argument("--key", type=str, default="text")
    parser.add_argument("--tokenizer", type=str, default="output/weights/llama2-7b")
    args = parser.parse_args()
    root = args.root
    split = f"{args.split}[:{args.limit}]"
    tokenizer_path = args.tokenizer
    tokenizer_name = tokenizer_path.split("/")[-1]
    dataset_name = args.dataset
    num_files = args.num_files
    key = args.key
    print(f"{args=}, {split=}")

    dest = Path(root) / dataset_name / tokenizer_name / f"{split}_tokenized"
    dest.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    dataset = get_text_pretrain_partial_dataset(
        dataset_name, root=root, split=split, key=key
    )

    indexes = range(0, len(dataset), len(dataset) // num_files)
    partial_datasets = [
        get_dataset_slice(dataset, start, end)
        for start, end in zip(indexes[:-1], indexes[1:])
    ]
    partial_total = len(dataset) // num_files
    eos_id = tokenizer.eos_token_id
    print(f"Executing {num_files} files in parallel")
    with concurrent.futures.ProcessPoolExecutor(num_files) as executor:
        for _ in executor.map(write_tokenized_dataset_in_parallel, range(num_files)):
            pass
    print("Files saved to disk at ", dest)

    """
    python -m scripts.tokenize_parallel --num_files 10 --root output/datasets --dataset c4 --split train --limit 1% --tokenizer output/weights/llama2-7b
    python -m scripts.tokenize_parallel --num_files 10 --root output/datasets --dataset wikipedia --split train --limit 100% --tokenizer output/weights/llama2-7b
    """
