import logging
import os
import random
import sys
import time
from functools import partial

import torch
from datasets import load_dataset
from omegaconf import DictConfig
from torch.utils.data import DataLoader, DistributedSampler
from torchtune import config, training, utils
from torchtune.data import Message
from torchtune.modules import TransformerDecoder
from torchtune.modules.tokenizers import ModelTokenizer
from tqdm import tqdm

from projects.sc.download import CACHE_DIR
from projects.sc.few_shot_prompt import gsm8k
from projects.sc.generate_sc import generate_lm_sc
from research.llm.collate import collate_self_consistency
from research.utils import distrib
from research.utils.logging import setup_logger
from research.utils.utils import seed_all

log = logging.getLogger(__file__)


@config.parse
def main(cfg: DictConfig) -> None:
    level = cfg.get("level", "DEBUG")
    setup_logger(level)
    config.log_config(recipe_name="SCEvalRecipe", cfg=cfg)
    log.info("Start")
    distrib.distrib_setup("cuda:nccl,cpu:gloo")
    world_size, rank = distrib.get_world_size_and_rank()
    log.info(f"{world_size=}, {rank=}")
    is_rank0 = rank == 0
    if rank > 0:
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    seed = cfg.get("seed", 42)
    seed_all(seed)

    device = utils.get_device(device=cfg.device)
    dtype = training.get_dtype(dtype=cfg.dtype, device=device)

    # Setup model
    """
    `https://pytorch.org/tutorials/recipes/recipes/module_load_state_dict_tips.html`
    Model initialization has some important considerations:
       a. To minimize GPU peak memory, we initialize the model on meta device with
          the right dtype
       b. All ranks calls `load_state_dict` without peaking CPU RAMs since
          full state dicts are loaded with `torch.load(mmap=True)`
    """
    log.info("Setting up model")
    t0 = time.perf_counter()
    with training.set_default_dtype(dtype), torch.device("meta"):  # torch.device() context manager
        model: TransformerDecoder = config.instantiate(cfg.model)

    compile = cfg.get("compile", True)
    if compile:
        training.compile_model(model, verbose=is_rank0)

    checkpointer = config.instantiate(cfg.checkpointer)
    model_state_dict = checkpointer.load_checkpoint()["model"]  # mmap=True

    log.info("Loading weights to model and set to evaluation mode.")
    model.load_state_dict(model_state_dict, assign=True)
    # RoPE is not covered in state dict
    for m in model.modules():
        if hasattr(m, "rope_init"):
            m.rope_init()

    # only suits small models
    model = model.to(device=rank, dtype=dtype)
    model.eval()
    torch.distributed.barrier()
    t1 = time.perf_counter() - t0
    log.info(f"Model loaded in {t1:.2f} s")

    # Setup tokenizer
    log.info("Loading tokenizer")
    tokenizer: ModelTokenizer = config.instantiate(cfg.tokenizer)
    log.info(f"{tokenizer=}")

    log.info("Loading dataset")
    dataset_name = cfg.get("dataset_name", "gsm8k")

    if dataset_name == "gsm8k":
        dataset = gsm8k(cfg)

    elif dataset_name == "gsm8k-before":
        cfg.before = True
        dataset = gsm8k(cfg)
    else:
        raise

    shuffle = False
    batch_size = cfg.get("batch_size", 1)  # only supports batch size of 1
    n_paths = cfg.get("n_paths", 8)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=seed
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=max(batch_size // n_paths, 1),
        sampler=sampler,
        # dropping last avoids shape issues with compile + flex attention
        drop_last=True,
        collate_fn=partial(
            collate_self_consistency,
            n_paths=n_paths,
        ),
    )

    # Sampling
    log.info("Sampling")
    stop_tokens = tokenizer.stop_tokens
    outs = {}
    outs[rank] = []
    top_p = cfg.get("top_p", None)
    top_k = cfg.get("top_k", None)
    total = 1e9
    pbar = tqdm(total=min(len(dataloader), total))
    for idx, batch in enumerate(dataloader):
        tokens = batch["tokens"].to(torch.device(f"cuda:{rank}"))
        out = generate_lm_sc(
            model,
            tokens,
            temperature=cfg.temperature,
            max_len=3072,
            stop_tokens=stop_tokens,
            top_k=top_k,
            top_p=top_p,
        )
        # "tokens": tokens,
        # "mask": mask,
        # "question": q,
        # "answer": a,
        # "prompt": prompt,
        # "final_answer": final_answer,
        for k, v in batch.items():
            if not isinstance(v, torch.Tensor):
                out[k] = v
        out["seed"] = seed
        out["dataset_id"] = dataset_name
        out["model_id"] = cfg.model_id
        out["rank"] = rank
        out["world_size"] = world_size

        # Save the results
        log.debug("Save Results")
        features_dir = os.path.join(cfg.output_dir, "features", cfg.model_id)
        os.makedirs(features_dir, exist_ok=True)
        results_path = os.path.join(
            features_dir, f"{idx}_{rank}-{world_size}_{dataset_name}_results_{seed}.pt"
        )
        torch.save(out, results_path)

        pbar.update(1)
        if idx > total:
            break

    torch.distributed.barrier()
    distrib.distrib_cleanup()
    log.info("End")


if __name__ == "__main__":
    sys.exit(main())
