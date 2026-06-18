import logging
import random
from typing import Any

import torch
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from research.utils.logging import setup_logger
from torchtune import config
from torchtune.data import Message
from torchtune.modules.tokenizers import ModelTokenizer
from tqdm import tqdm

from projects.sc.download import CACHE_DIR

log = logging.getLogger(__file__)


def gsm8k(cfg: DictConfig):
    level = cfg.get("level", "DEBUG")
    setup_logger(level)
    # Setup tokenizer
    log.debug("Loading tokenizer")
    tokenizer: ModelTokenizer = config.instantiate(cfg.tokenizer)
    log.debug(f"{tokenizer=}")

    log.debug("Loading dataset")
    ds = load_dataset("openai/gsm8k", "main", cache_dir=CACHE_DIR)
    ds_any: Any = ds
    test_ds = ds_any["test"]
    train_ds = ds_any["train"]
    q = test_ds["question"][0]
    a = test_ds["answer"][0]
    log.debug(f"{ds=}")
    log.debug(f"\n{q=}\n{a=}")

    # Building few shots
    log.debug("Building few shots")
    n_shots = cfg.get("n_shots", 8)
    random_shots = [random.randrange(0, len(train_ds)) for _ in range(n_shots)]
    qs = [train_ds["question"][r] for r in random_shots]
    ans = [train_ds["answer"][r] for r in random_shots]
    few_shot = [
        Message(
            role="system",
            content="As an expert problem solver solve step by step the following mathematical questions.",
        )
    ]
    before = cfg.get("before", False)
    for q, a in zip(qs, ans, strict=False):
        aa = a.split("\n#### ")
        if before:
            str_a = "The answer is: " + aa[1].strip() + ".\n" + aa[0].strip()
        else:
            str_a = aa[0].strip() + "\nThe answer is: " + aa[1].strip() + "."
        few_shot.append(Message(role="user", content=q))
        few_shot.append(Message(role="assistant", content=str_a))

    log.info(f"{few_shot=}")

    # Tokenizing dataset
    log.debug("Tokenizing dataset")
    dataset = []
    total = len(test_ds)
    pbar = tqdm(total=total, desc="Tokenizing dataset")
    for i in range(total):
        q = test_ds["question"][i]
        a = test_ds["answer"][i]
        prompt = [*few_shot]
        prompt.append(Message(role="user", content=q))
        tokens, mask = tokenizer.tokenize_messages(prompt)
        tokens = torch.tensor(tokens, dtype=torch.int)
        final_answer = a.split("\n#### ")[-1].strip()
        dataset.append({
            "tokens": tokens,
            "mask": mask,
            "question": q,
            "answer": a,
            "prompt": prompt,
            "final_answer": final_answer,
        })
        pbar.update(1)

    return dataset


if __name__ == "__main__":
    cfg_path = "projects/sc/configs/llama3_1_8b_config.yaml"
    cfg = OmegaConf.load(cfg_path)
    gsm8k(cfg)
