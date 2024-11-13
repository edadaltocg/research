import base64
import gc
import json
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import regex

# from tensordict import TensorDict, MemoryMappedTensor
import torch
import torch.utils.data
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, IterableDataset
from torchvision.transforms import v2
from tqdm import tqdm

from dnn.tokenizer import Tokenizer


class DummyDataset(Dataset):
    def __init__(self, n: int = 1000, dim: int = 768, num_classes=2, *args, **kwargs):
        self.n = n
        self.dim = dim
        self.num_classes = num_classes

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "x": torch.randn(self.dim),
            "y": torch.randint(0, self.num_classes, (1,)),
        }


class DummyImageDataset(Dataset):
    def __init__(self, n: int = 1000, size: int = 224, *args, **kwargs):
        self.n = n
        self.size = size

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return torch.randn(self.size, 3, self.size, self.size)


class DummyImageClassificationDataset(Dataset):
    def __init__(
        self, n: int = 1000, num_classes: int = 1000, size: int = 224, *args, **kwargs
    ):
        self.n = n
        self.num_classes = num_classes
        self.size = size

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "x": torch.randn(3, self.size, self.size),
            "y": torch.randint(0, 1000, (1,)).item(),
        }


class DummyTextDataset(Dataset):
    def __init__(
        self, n: int = 1000, max_len: int = 100, vocab_size: int = 1000, *args, **kwargs
    ):
        self.n = n
        self.max_len = max_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return torch.randint(0, self.vocab_size, (self.max_len,))


class DummyImageTextDataset(Dataset):
    def __init__(
        self,
        n: int = 1000,
        img_size: int = 224,
        max_len: int = 100,
        vocab_size: int = 1000,
        *args,
        **kwargs,
    ):
        self.n = n
        self.img_size = img_size
        self.max_len = max_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return (
            torch.randn(3, self.img_size, self.img_size),
            torch.randint(0, self.vocab_size, (self.max_len,)),
        )


class DiskReaderDataset(Dataset):
    def __init__(self, root: str) -> None:
        Path(root).is_dir(), f"Path {root} is not a directory"
        self.root = root
        self.list_of_files = list(sorted(Path(root).glob("*")))

    def __getitem__(self, index):
        file_name = self.list_of_files[index]
        return torch.load(file_name)

    def __len__(self):
        return len(self.list_of_files)


class InMemoryDataset(Dataset):
    def _init__(self, data: list[Tensor]) -> None:
        self.data = data

    def __getitem__(self, index) -> Tensor:
        return self.data[index]

    def __len__(self):
        return len(self.data)


class PreTransformImageClassificationDataset(Dataset):
    def __init__(self, dataset, transform=None, x_key="img", y_key="label"):
        self.dataset = dataset
        self.transform = transform
        self.x_key = x_key
        self.y_key = y_key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx][self.x_key]
        label = self.dataset[idx][self.y_key]
        if self.transform:
            img = self.transform(img)
        return {"x": img, "y": label}


class MemoryMappedDataset(Dataset):
    """Pre-load the entire dataset into a memory-mapped tensor.

    Limitations:
        The tensors should have always same size (no variable length allowed).
    """

    def __init__(
        self, root: str, transform: Optional[Callable] = None, x_key="x", y_key="y"
    ):
        self.memmaps = {}
        self.x_key = x_key
        self.y_key = y_key
        self.transform = transform
        metadata = torch.load(Path(root) / "metadata.pt")
        for d in metadata:
            self.memmaps[d["key"]] = np.memmap(
                d["filename"], dtype=d["dtype"], mode="r+", shape=d["shape"]
            )
        self.size = metadata["size"]

    def __getitem__(self, index):
        data = {k: torch.as_tensor(v[index]) for k, v in self.memmaps.items()}
        if self.transform is not None:
            return {"x": self.transform(data[self.x_key]), "y": data[self.y_key]}
        return data

    def __len__(self):
        return self.size

    def __del__(self):
        for k, v in self.memmaps.items():
            del v
            gc.collect()

    @staticmethod
    def from_dataloader(
        dataloader, root, transform: Optional[Callable] = None, x_key="x", y_key="y"
    ):
        Path(root).mkdir(parents=True, exist_ok=True)
        memmaps = {}
        size = len(dataloader.dataset)  # type: ignore

        # consume dataset in batches and save to memmap
        b = 0
        for batch in tqdm(dataloader, desc="Saving dataset to memmap"):
            for k, v in batch.items():
                v = torch.as_tensor(v)
                v = v.cpu().numpy()
                v_size = v.shape[1:]
                v_dtype = v.dtype
                if k not in memmaps:
                    memmaps[k] = np.memmap(
                        Path(root) / f"{k}.memmap",
                        dtype=v_dtype,
                        mode="w+",
                        shape=(size, *v_size),
                    )
                memmaps[k][b : b + len(v)] = v
            b += len(v)  # type: ignore

        # force garbage collection to free up memory
        gc.collect()
        for k in memmaps:
            memmaps[k].flush()

        # metadata: size, shapes, dtypes, keys, etc
        metadata = [
            {
                "key": k,
                "dtype": v.dtype,
                "shape": v.shape,
                "size": size,
                "filename": Path(root) / f"{k}.memmap",
            }
            for k, v in memmaps.items()
        ]
        metadata = metadata
        torch.save(metadata, Path(root) / "metadata.pt")
        metadata = [{k: str(v) for k, v in d.items()} for d in metadata]
        with open(Path(root) / "metadata.json", "w") as f:
            json.dump(metadata, f)

        return MemoryMappedDataset(root, transform, x_key, y_key)


class TokenizeTextDataset(Dataset):
    def __init__(self, dataset, tokenizer: Tokenizer, key="text"):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.key = key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> list[int]:
        text = self.dataset[self.key][idx].strip()
        tokens = self.tokenizer.raw_encode(text, bos=True, eos=False)
        return tokens


class IterableTokenizeTextDataset(IterableDataset):
    def __init__(self, dataset, tokenizer: Tokenizer, key="text"):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.key = key

    def __iter__(self):
        for idx in range(len(self.dataset)):
            text = self.dataset[self.key][idx]
            tokens = self.tokenizer.raw_encode(text, bos=False, eos=False)
            yield tokens

    def __len__(self):
        return len(self.dataset)


class MemoryMapped1DDataset(Dataset):
    MAX_FILE_SIZE = 1024**2
    MAX_NUM_TOKENS = 1024**2 // 2

    def __init__(self, root: str, chunk_size: int):
        metadata = torch.load(Path(root) / "metadata.pt")
        self.size = 0
        self.memmaps = {}
        begin = 0
        end = 0
        for d in metadata:
            new_size = d["shape"][0] // chunk_size * chunk_size
            if new_size == 0:
                continue
            end = begin + d["shape"][0] // chunk_size
            self.size += d["shape"][0] // chunk_size
            data = np.memmap(
                d["filename"], dtype=d["dtype"], mode="r+", shape=d["shape"]
            )
            self.memmaps[(begin, end)] = data[:new_size].reshape(-1, chunk_size)
            begin = end

        # self.memmap = np.memmap(metadata["filename"], dtype=metadata["dtype"], mode="r+", shape=metadata["shape"])
        # self.size = metadata["size"]
        # self.chunk_size = chunk_size
        # if chunk_size is not None:
        #     self.size = self.memmap.shape[0] // chunk_size
        #     self.memmap = self.memmap[: self.size * chunk_size]
        #     self.memmap.reshape(-1, chunk_size)

    def __getitem__(self, index) -> Dict[str, Tensor]:
        # find appropriate key
        for (begin, end), memmap in self.memmaps.items():
            if begin <= index < end:
                data = memmap[index - begin].tolist()
                data = torch.as_tensor(data, dtype=torch.long)
                return {"x": data}
        raise IndexError

    def __len__(self):
        return self.size

    @staticmethod
    def from_dataloader(dataloader, root, dtype=np.uint16):
        Path(root).mkdir(parents=True, exist_ok=True)

        # consume dataset in batches and save to memmap
        metadata = []
        chunk = np.ones((1024**3 // 2,), dtype=dtype)  # 1GB (536870912,)
        curr_size = 0
        idx = 0
        all_size = 0
        pbar = tqdm(dataloader, desc="Saving dataset to memmap")
        for _, batch in enumerate(pbar):
            batch_arr = np.array(batch, dtype=dtype)

            if curr_size + len(batch) >= len(chunk):
                batch_arr_1 = batch_arr[: len(chunk) - curr_size]
                batch_arr_2 = batch_arr[len(chunk) - curr_size :]
                chunk[curr_size:] = batch_arr_1
                fp = np.memmap(
                    Path(root) / f"data-{idx:03}.memmap",
                    dtype=dtype,
                    mode="w+",
                    shape=(len(chunk),),
                )
                fp[:] = chunk
                fp.flush()
                print(f"Saved chunk to file: data-{idx:03}.memmap")
                metadata.append(
                    {
                        "dtype": dtype,
                        "shape": (len(chunk),),
                        "size": len(chunk),
                        "filename": f"data-{idx:03}.memmap",
                    }
                )
                idx += 1
                chunk[: len(batch_arr_2)] = batch_arr_2
                curr_size = len(batch_arr_2)
            else:
                chunk[curr_size : curr_size + len(batch_arr)] = batch_arr
                curr_size += len(batch_arr)

            all_size += len(batch_arr)
            pbar.update()
            pbar.set_postfix({"size": all_size})

        # save the last chunk
        chunk = chunk[:curr_size]
        fp = np.memmap(
            Path(root) / f"data-{idx:03}.memmap",
            dtype=dtype,
            mode="w+",
            shape=(len(chunk),),
        )
        fp[:] = chunk
        fp.flush()
        print(f"Saved chunk to file: data-{idx:03}.memmap")
        metadata.append(
            {
                "dtype": dtype,
                "shape": (len(chunk),),
                "size": len(chunk),
                "filename": Path(root) / f"data-{idx:03}.memmap",
            }
        )

        torch.save(metadata, Path(root) / "metadata.pt")
        metadata = [{k: str(v) for k, v in d.items()} for d in metadata]
        with open(Path(root) / "metadata.json", "w") as f:
            json.dump(metadata, f)

        # collect all the chunks
        all_metadata = {
            "size": all_size,
            "dtype": dtype,
            "shape": (all_size,),
            "filenames": [],
        }
        for d in metadata:
            all_metadata["filenames"].append(d["filename"])
        torch.save(all_metadata, Path(root) / "all_metadata.pt")
        all_metadata = {k: str(v) for k, v in all_metadata.items()}
        with open(Path(root) / "all_metadata.json", "w") as f:
            json.dump(all_metadata, f)


class PreTrainIterableDataset(IterableDataset): ...


class ImageClassificationDataset(Dataset):
    """Chunked Image Classification Dataset.
    Each chunk is a list of tuples containing tensor images and tensor labels the data among the chunks is not shuffled
    the chunks are named like 'chunk-idx1-idx2.pt', for example: 'chunk-0-1000.pt' contains the first 1000 images
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable[[Image.Image], Tensor]] = None,
        target_transform: Optional[Callable[[int], int]] = None,
    ) -> None:
        assert Path(root).is_dir(), f"Path {root} is not a directory"
        self.list_of_chunks = list(Path(root).glob("chunk*.pt"))
        self._len = sum([len(torch.load(chunk)) for chunk in self.list_of_chunks])
        self.list_of_start_end_indices = [
            tuple(chunk.stem.split("-")[1:]) for chunk in self.list_of_chunks
        ]
        # separate the names of the chunks into a list of tuples (start, end)
        self.mapping = {
            idx: chunk
            for idx, chunk in zip(self.list_of_start_end_indices, self.list_of_chunks)
        }
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        # find the chunk that contains the idx
        for (start, end), chunk in self.mapping.items():
            if int(start) <= idx < int(end):
                break
        chunk = torch.load(chunk)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return {"x": ..., "y": ...}


def make_image_pre_processor(
    resize_size: Optional[int | tuple[int, int]] = None,
) -> Callable:
    if resize_size is None:
        return v2.Compose([v2.PILToTensor()])
    return v2.Compose([v2.PILToTensor(), v2.Resize(resize_size)])


def make_image_processor(
    # ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
    # Grayscale(num_output_channels=3)
    img_size: int | tuple[int, int],
    resize_size: Optional[int | tuple[int, int]] = None,
    mean: Sequence[float] = [0.485, 0.456, 0.406],
    std: Sequence[float] = [0.229, 0.224, 0.225],
    training: bool = False,
    pre_transform=None,
) -> Callable[[Image.Image], Tensor]:
    """Create a torchvision.transforms.Compose function for image processing.
    pre_transform = [v2.PILToTensor(), v2.Resize(resize_size)]
    """
    if resize_size is None:
        resize_size = img_size
    if training:
        transforms = [
            v2.RandomResizedCrop(img_size, antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.AutoAugment(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean, std),
        ]
    else:
        transforms = [
            v2.Resize(resize_size),
            v2.CenterCrop(img_size),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean, std),
        ]
    if pre_transform:
        transforms = pre_transform + transforms
    fn = v2.Compose(transforms)
    return fn


def img2base64_str(file_name: str) -> str:
    with open(file_name, "rb") as img_file:
        base64_bytes = base64.b64encode(img_file.read())

    return base64_bytes


def preprocess_text(text: str) -> list[str]:
    gpt2_pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    gpt4_pattern = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"
    compiled_pattern = regex.compile(gpt4_pattern)
    # split the text up into text ch
    text_chunks = regex.findall(compiled_pattern, text)
    if len(text_chunks) == 0:
        text_chunks = [" "]
    # input text preprocessing
    return text_chunks
