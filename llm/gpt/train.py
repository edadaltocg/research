import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data

from datasets import load_dataset, load_from_disk
from dnn.data import MemoryMapped1DDataset, TokenizeTextDataset
from dnn.tokenizer import Tokenizer, train_sp_tokenizer_from_iterator
from dnn.trainer import standard_trainer
from gpt.model import GPT
from utils.utils import collate_flat


def train_gpt2_wiki(
    vocab_size=50304,
    embed_dim=768,
    max_seq_len=1024,
    num_heads=12,
    num_layers=12,
    num_epochs=300,
    batch_size=32,
    train_tokenizer=False,
    download=False,
    memmap=False,
):
    if download:
        dataset = load_dataset(
            "wikitext", "wikitext-2-raw-v1", num_proc=os.cpu_count() - 1
        )  # type: ignore
        dataset.save_to_disk("output/datasets/wikitext")  # type: ignore

    if train_tokenizer:
        dataset = load_from_disk("output/datasets/wikitext")
        print(dataset)
        train_sp_tokenizer_from_iterator(
            iter(dataset["train"]["text"]), prefix="gpt_wiki", vocab_size=vocab_size
        )

    if memmap:
        tokenizer = Tokenizer("output/tokenizers/gpt_wiki.model")
        for split in ["train", "validation"]:
            ds = load_from_disk("output/datasets/wikitext")[split]
            dataset = TokenizeTextDataset(ds, tokenizer, key="text")  # list[int]
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=128,
                num_workers=8,  # type: ignore
                collate_fn=collate_flat,
            )
            MemoryMapped1DDataset.from_dataloader(
                dataloader, f"output/datasets/wikitext/{split}", dtype=np.uint16
            )

    def get_model():
        model = GPT(
            # **gpt2_tiny_config,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            max_seq_len=max_seq_len,
            num_heads=num_heads,
            num_layers=num_layers,
        )
        return model

    def get_optimizer(model, lr=4e-4):
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-5, weight_decay=0.1
        )
        return optimizer

    def get_lr_scheduler(optimizer, warmup_steps=5000):
        return

    def _get_dataset(split):
        dataset = MemoryMapped1DDataset(
            f"output/datasets/wikitext/{split}", max_seq_len + 1
        )
        return dataset

    def get_train_dataset():
        return _get_dataset("train")

    def get_eval_dataset():
        return _get_dataset("validation")

    def get_loss(model, batch):
        x = batch["x"]
        y = x[..., 1:].clone().contiguous().view(-1)
        x = x[..., :-1].contiguous()
        logits = model(x)
        logits_loss = logits.view(-1, logits.size(-1))
        loss = F.cross_entropy(logits_loss, y, reduction="sum")
        ppl = torch.exp(loss / y.size(0))
        return loss

    standard_trainer(
        get_model,
        get_train_dataset,
        get_eval_dataset,
        get_optimizer,
        get_loss,
        None,
        num_epochs,
        batch_size=batch_size,
        model_prefix="gpt2_wiki",
        gradient_clip=1.0,
    )
