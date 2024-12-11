import sys
import random

import torch
from omegaconf import DictConfig

from torchtune import config, training, utils
from torchtune.modules import TransformerDecoder
from torchtune.modules.tokenizers import TikTokenBaseTokenizer

from projects.sc.eval import flexible_extract, strict_match
from projects.sc.generate_sc import generate_lm_sc
from research.utils.logging import setup_logger
import logging
from datasets import load_dataset

from research.utils.utils import seed_all

log = logging.getLogger(__file__)
setup_logger(logging.INFO)

PREAMBLE = (
    """As an expert problem solver solve step by step the following mathematical questions."""
)
# The default gsm8k prompt from the CoT paper
# https://arxiv.org/pdf/2201.11903.pdf page 35.
FEW_SHOT = "Question: {question}\nAnswer: {answer}\n\n"
TEMPLATE = "Question: {question}\nAnswer:"


@config.parse
def main(cfg: DictConfig) -> None:
    seed_all(cfg.seed)

    config.log_config(recipe_name="SCEvalRecipe", cfg=cfg)
    device = utils.get_device(device=cfg.device)
    dtype = training.get_dtype(dtype=cfg.dtype, device=device)

    with training.set_default_dtype(dtype), device:
        model: TransformerDecoder = config.instantiate(cfg.model)

    checkpointer = config.instantiate(cfg.checkpointer)
    model_state_dict = checkpointer.load_checkpoint()["model"]
    log.info("Loading weights to model and set to evaluation mode.")
    model.load_state_dict(model_state_dict)
    model = model.to(device=device, dtype=dtype)
    model.eval()

    log.info("Loading tokenizer")
    tokenizer: TikTokenBaseTokenizer = config.instantiate(cfg.tokenizer)

    log.info("Loading dataset")
    ds = load_dataset("openai/gsm8k", "main")
    q = ds["test"]["question"][0]
    a = ds["test"]["answer"][0]
    log.info(f"{ds=}")
    log.info(f"\n{q=}\n{a=}")

    n_shots = cfg.get("n_shots", 5)
    random_shots = [random.randrange(0, len(ds["train"])) for _ in range(n_shots)]
    log.info(f"{random_shots=}")
    qs = [ds["train"]["question"][r] for r in random_shots]
    ans = [ds["train"]["answer"][r] for r in random_shots]

    few_shot = "".join([FEW_SHOT.format(question=q, answer=a) for q, a in zip(qs, ans)])
    prompt = f"{PREAMBLE}\n\n{few_shot}{TEMPLATE.format(question=q)}"
    log.info(f"\nprompt={prompt}")

    out = generate_lm_sc(
        model,
        tokenizer,
        prompt,
        n_paths=5,
        temperature=cfg.temperature,
        max_new_tokens=256,
        stop_tokens=[
            tokenizer.eos_id,
            *tokenizer.encode("Question", add_bos=False, add_eos=False),
        ],
    )

    # Filter response
    model_ans = [strict_match(ans).split("#### ")[-1].rstrip() for ans in out["answers"]]
    correct_ans = a.split("#### ")[-1].rstrip()
    log.info(f"{model_ans=},\n{correct_ans=}")


if __name__ == "__main__":
    sys.exit(main())
