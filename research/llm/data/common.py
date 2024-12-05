import logging
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from datasets import load_from_disk
from torch.utils.data import ConcatDataset, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

log = logging.getLogger(__name__)
ROOT = Path("output/datasets")


def get_dataset_offline(root=ROOT / "c4", split="train[:1%]"):
    dataset = load_from_disk(str(Path(root) / split))
    return dataset


class TextPretrainPartialDataset(Dataset):
    def __init__(self, root: str | Path, split: str, key: str = "text") -> None:
        self.root = str(root)
        self.split = str(split)
        self.key = key
        self.dataset = self.build_dataset()
        print(f"{self.dataset=}")

    def build_dataset(self) -> Dataset:
        return get_dataset_offline(self.root, self.split)

    def __getitem__(self, index: int) -> list[str]:
        elem = self.dataset[index]
        text = elem[self.key]
        # text = preprocess_text(text)
        return text

    def __len__(self) -> int:
        return len(self.dataset)


def get_text_pretrain_partial_dataset(
    dataset_name: str,
    root: str | Path = "output/datasets",
    split: str = "train[:1%]",
    key: str = "text",
) -> Dataset:
    return TextPretrainPartialDataset(
        root=Path(root) / dataset_name, split=split, key=key
    )


def get_text_pretrain_dataset(root=ROOT):
    datasets = [
        get_text_pretrain_partial_dataset("c4", root, split="train[:1%]"),
        get_text_pretrain_partial_dataset("wikipedia", root, split="train[:1%]"),
        get_text_pretrain_partial_dataset("the_stack", root, split="train[:1%]"),
    ]
    dataset = ConcatDataset(datasets)
    return dataset


def write_tokenized_dataset(dest="output/datasets/mypile_tokenized"):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained("output/weights/llama2-7b")
    assert tokenizer.is_fast
    eos_tok = tokenizer.eos_token
    eos_id = tokenizer.eos_token_id
    log.info(f"{eos_tok=}, {eos_id=}")

    dataset = get_text_pretrain_dataset()
    total = len(dataset)

    def tokenizer_fn(text: list[str]):
        ids = tokenizer(text, padding=False, truncation=False)
        ids = ids.input_ids + [eos_id]
        return ids

    generator = map(tokenizer_fn, dataset)
    batch_len = 0
    i = 0
    batch = []
    with open(dest / "tokens.bin", "wb") as f:
        # 1h
        for tokens in tqdm(generator, total=total):
            batch += tokens
            batch_len += len(tokens)
            if batch_len >= 1024 * 1024:
                batch_len = 0
                batch = []

                print(f"Writing batch {i}", flush=True)
                # list of int to uint16 array
                tokens = np.array(batch, dtype=np.uint16)
                f.write(tokens.tobytes())
                f.flush()


class TokenizedPreTrainDataset(Dataset):
    def __init__(
        self,
        filename="output/datasets/mypile_tokenized/tokens.bin",
        block_size: int = 4096,
        padding_value: int = 0,
    ) -> None:
        self.filename = filename
        self.buffer_size = block_size
        self.block_size = block_size
        self.original_dt = np.uint16
        self.padding_value = padding_value

    def __getitem__(self, index: int) -> torch.Tensor:
        with open(self.filename, "rb") as f:
            f.seek(index * self.buffer_size)
            buffer = f.read(self.buffer_size)
        buffer = np.frombuffer(buffer, dtype=self.original_dt)
        if len(buffer) < self.block_size:
            buffer = np.pad(
                buffer,
                (0, self.block_size - len(buffer)),
                mode="constant",
                constant_values=self.padding_value,
            )
        tensor = torch.from_numpy(np.int64(buffer)).reshape(1, self.block_size)
        return tensor

    def __len__(self) -> int:
        return int(Path(self.filename).stat().st_size / self.buffer_size)


class ConcatDatasetsWithProbabilities(Dataset):
    def __init__(self, datasets: list[Dataset], probabilities: list[float]) -> None:
        self.datasets = datasets
        self.probabilities = [p / sum(probabilities) for p in probabilities]
        self.cumulative_probabilities = np.cumsum(self.probabilities)

    def __getitem__(self, index: int) -> torch.Tensor:
        p = np.random.rand()
        dataset_idx = np.searchsorted(self.cumulative_probabilities, p, side="right")
        dataset = self.datasets[dataset_idx]
        index = np.random.randint(len(dataset))
        return dataset[index]

    def __len__(self) -> int:
        return sum(len(d) for d in self.datasets)


def test_read_tokenized_dataset():
    dataset = TokenizedPreTrainDataset()
    for i in range(len(dataset)):
        print(dataset[i].shape)
        break


def build_pre_train_dataset(
    dataset_names_splits_and_proportions: dict[tuple[str, str], float],
    root=ROOT,
    tokenizer="output/weights/llama2-7b",
    block_size=4096,
    padding_value=0,
):
    tokenizer_name = Path(tokenizer).name
    datasets = []
    for dataset_name, split in dataset_names_splits_and_proportions:
        path = Path(root) / dataset_name / tokenizer_name / f"{split}_tokenized"
        filelist = list(path.glob("*.bin"))
        tpm_datasets = [
            TokenizedPreTrainDataset(str(f), block_size, padding_value=padding_value)
            for f in filelist
        ]
        tmp_dataset = ConcatDataset(tpm_datasets)
        datasets.append(tmp_dataset)
    dataset = ConcatDatasetsWithProbabilities(
        datasets, list(dataset_names_splits_and_proportions.values())
    )

    # min_length = min(len(d) for d in datasets)
    # sum_proportions = sum(dataset_names_splits_and_proportions.values())
    # for idx, proportion in enumerate(dataset_names_splits_and_proportions.values()):
    #     datasets[idx] = torch.utils.data.Subset(
    #         datasets[idx],
    #         torch.randperm(min_length)[: int(min_length * proportion / sum_proportions)]
    #         .numpy()
    #         .tolist(),
    #     )
    # dataset = ConcatDataset(datasets)

    return dataset


if __name__ == "__main__":
    import fire

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    fire.Fire(
        {
            "write_tokenized_dataset": write_tokenized_dataset,
            "read_tokenized_dataset": test_read_tokenized_dataset,
        }
    )
