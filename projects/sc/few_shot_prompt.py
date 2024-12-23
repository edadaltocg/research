import sys
import logging
import random

import torch
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from torchtune import config
from torchtune.data import Message
from torchtune.modules.tokenizers import ModelTokenizer
from tqdm import tqdm

from projects.sc.download import CACHE_DIR
from research.utils.logging import setup_logger

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
    q = ds["test"]["question"][0]
    a = ds["test"]["answer"][0]
    log.debug(f"{ds=}")
    log.debug(f"\n{q=}\n{a=}")

    # Building few shots
    log.debug("Building few shots")
    n_shots = cfg.get("n_shots", 8)
    random_shots = [random.randrange(0, len(ds["train"])) for _ in range(n_shots)]
    qs = [ds["train"]["question"][r] for r in random_shots]
    ans = [ds["train"]["answer"][r] for r in random_shots]
    few_shot = [
        Message(
            role="system",
            content="As an expert problem solver solve step by step the following mathematical questions.",
        )
    ]
    before = cfg.get("before", False)
    for q, a in zip(qs, ans):
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
    total = len(ds["test"])
    pbar = tqdm(total=total, desc="Tokenizing dataset")
    for i in range(total):
        q = ds["test"]["question"][i]
        a = ds["test"]["answer"][i]
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
