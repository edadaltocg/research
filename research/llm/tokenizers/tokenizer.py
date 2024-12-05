import io
import os
import string
from pathlib import Path

import torch
from sentencepiece import SentencePieceProcessor, SentencePieceTrainer
from transformers.modeling_utils import json


class Tokenizer:
    def __init__(self, path) -> None:
        assert Path(path).exists(), f"{path} does not exist"
        self._model = SentencePieceProcessor(path)  # type: ignore
        self.bos_id = self._model.bos_id()
        self.eos_id = self._model.eos_id()
        self.pad_id = self._model.pad_id()
        self.unk_id = self._model.unk_id()
        self.vocab_size = self._model.vocab_size()
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.unk_token = "<unk>"
        self.pad_token = "<pad>"

    def encode(
        self, string: str | list[str], bos: bool = True, eos: bool = False, max_len=None
    ) -> torch.Tensor:
        if isinstance(string, str):
            return self.batch_encode([string], bos=bos, eos=eos, max_len=max_len)[0]
        return self.batch_encode(string, bos=bos, eos=eos, max_len=max_len)

    def batch_encode(
        self, strings: list[str], bos=True, eos=False, max_len=None
    ) -> torch.Tensor:
        tokens = self._model.encode(strings, out_type=int)  # type: ignore
        if bos:
            tokens = [[self.bos_id] + t for t in tokens]
        if eos:
            tokens = [t + [self.eos_id] for t in tokens]
        if max_len is not None:
            tokens = [t[:max_len] for t in tokens]
            tokens = [t + [self.pad_id] * (max_len - len(t)) for t in tokens]
        else:
            max_len = max(len(t) for t in tokens)
            tokens = [t + [self.pad_id] * (max_len - len(t)) for t in tokens]

        return torch.tensor(tokens, dtype=torch.int)

    def decode(self, tensor: torch.Tensor) -> str | list[str]:
        tokens = tensor.tolist()
        return self._model.decode(tokens)  # type: ignore

    def batch_decode(self, tensor: torch.Tensor) -> str | list[str]:
        return self.decode(tensor)

    def get_attn_mask(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor != self.pad_id).int()

    def raw_encode(
        self,
        string: str | list[str],
        bos: bool = True,
        eos: bool = False,
        out_type=int,
        *args,
        **kwargs,
    ):
        tokens = self._model.encode(string, out_type=out_type, *args, **kwargs)  # type: ignore
        if isinstance(string, str) and bos:
            tokens = [self.bos_id] + tokens
        elif isinstance(string, list) and bos:
            tokens = [[self.bos_id] + t for t in tokens]
        if isinstance(string, str) and eos:
            tokens = tokens + [self.eos_id]
        elif isinstance(string, list) and eos:
            tokens = [t + [self.eos_id] for t in tokens]
        return tokens


def train_sp_tokenizer_from_iterator(
    iterator, dest_path="output/tokenizers", prefix="sp", vocab_size=32768
):
    path = os.path.join(dest_path, f"{prefix}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ascii_chars = [c for c in string.printable]
    # filter out problematic ones
    ascii_chars = ascii_chars[:64]
    extra_chars = ascii_chars
    model = io.BytesIO()
    SentencePieceTrainer.train(  # type: ignore
        sentence_iterator=iterator,
        model_writer=model,
        vocab_size=vocab_size,
        model_type="bpe",
        shuffle_input_sentence=False,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        user_defined_symbols=set(extra_chars),  # all ascii characters
        allow_whitespace_only_pieces=1,
        split_digits=1,
        train_extremely_large_corpus=1,
    )
    with open(f"{path}.model", "wb") as f:
        f.write(model.getvalue())

    sp = SentencePieceProcessor()
    sp.load(f"{path}.model")  # type: ignore
    # sp.set_vocabulary(whitespaces, 9999)
    # save vocab
    vocab = {sp.id_to_piece(i): i for i in range(sp.get_piece_size())}
    with open(f"{path}_vocab.json", "w") as f:
        json.dump(vocab, f, indent=2)
    return sp
