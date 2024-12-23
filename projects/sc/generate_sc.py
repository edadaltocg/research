import itertools
import logging
import time
from typing import Any, Dict, Union

import torch
from torch import Tensor
from torchtune.modules import TransformerDecoder
from tqdm import tqdm

from projects.sc.inference import forward
from research.llm.sample import sample
from research.utils.logging import setup_logger

log = logging.getLogger(__file__)
setup_logger(logging.INFO)


def _log_metrics(model, total_time: float, prefill_time: float, tokens_per_second: float) -> None:
    """Logs the following metrics: total time for inference, tokens/sec,
    bandwidth achieved, and max memory allocated.

    Feel free to modify this function to log additional metrics.
    """
    model_size = sum([
        p.numel() * p.dtype.itemsize for p in itertools.chain(model.parameters(), model.buffers())
    ])
    log.info(f"Time for prefill: {prefill_time:.02f} sec")
    log.info(
        f"Total time for inference: {total_time:.02f} sec total, {tokens_per_second:.02f} tokens/sec"
    )
    log.info(f"Bandwidth achieved: {model_size * tokens_per_second / 1e9:.02f} GB/s")
    log.info(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def _update_stop_tokens_tracker(
    tokens: torch.Tensor, stop_tokens: torch.Tensor, stop_token_reached: torch.Tensor
) -> torch.Tensor:
    """Updates which sequences have reached a stop token."""
    # tokens: [bsz, 1]
    # stop_tokens: [num_stop_tokens]
    # stop_token_reached: [bsz]
    stop_token_reached_curr = torch.isin(tokens, stop_tokens).flatten()
    stop_token_reached |= stop_token_reached_curr
    return stop_token_reached


@torch.inference_mode()
def generate_lm_sc(
    model: TransformerDecoder,
    tokens: Tensor,
    *,
    max_len=4096,
    temperature=0.6,
    top_p=None,
    top_k=None,
    stop_tokens=None,
) -> Dict[str, Union[Tensor, Any]]:
    """The main entry point for generating tokens from a prompt."""

    bsz, prompt_len = tokens.size()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    max_new_tokens = max_len - prompt_len
    total_response_length = prompt_len + max_new_tokens

    # Setup KV cache
    with device:
        if model.caches_are_setup():
            model.reset_caches()
        if not model.caches_are_setup():
            model.setup_caches(
                batch_size=bsz,
                dtype=dtype,
                decoder_max_seq_len=total_response_length,
            )

    # Pre-allocate causal mask and input_pos
    causal_mask = torch.tril(
        torch.ones(
            size=(total_response_length, total_response_length),
            dtype=torch.bool,
            device=device,
        )
    ).unsqueeze(0)
    input_pos = (
        torch.arange(total_response_length).unsqueeze(0).to(dtype=torch.int32, device=tokens.device)
    )

    # Prefill step
    generated_tokens = torch.zeros(
        bsz, total_response_length, device=tokens.device, dtype=tokens.dtype
    )
    generated_tokens[:, :prompt_len] = tokens.clone()
    t0 = time.perf_counter()
    h, logits = forward(
        model,
        tokens,
        mask=causal_mask[:, :prompt_len],
        input_pos=input_pos[:, :prompt_len],
    )
    t_prefill = time.perf_counter() - t0

    tokens = sample(logits[:, -1], temperature=temperature, top_k=top_k, top_p=top_p)
    log.debug(f"{tokens.shape=}, {h.shape=}, {logits.shape=}")

    curr_pos = prompt_len

    generated_tokens[:, curr_pos] = tokens.squeeze()

    generated_logits = torch.zeros(
        bsz, total_response_length, logits.shape[-1], device=tokens.device, dtype=logits.dtype
    )
    generated_logits[:, :curr_pos] = logits

    generated_h = torch.zeros(
        bsz, total_response_length, h.shape[-1], device=tokens.device, dtype=h.dtype
    )
    generated_h[:, :curr_pos] = h

    # Stop token
    stop_token_reached = torch.zeros(bsz, dtype=torch.bool, device=device)
    stop_tokens = torch.tensor(stop_tokens, device=tokens.device, dtype=tokens.dtype)
    # everything in stop_token_mask starts as 1s, and we'll set them to 0 for sequences
    # that already hit a stop token
    stop_token_mask = torch.ones(
        (bsz, total_response_length), dtype=torch.int32, device=tokens.device
    )

    # Continue generating TODO: with greedy search?
    for _ in tqdm(range(max_new_tokens - 1)):
        # Update stop tokens mask
        stop_token_reached = _update_stop_tokens_tracker(tokens, stop_tokens, stop_token_reached)
        if stop_token_reached.all().item():
            log.debug(f"{stop_token_reached=}")
            break
        stop_token_mask[:, curr_pos] = ~stop_token_reached

        # Incremental decoding
        curr_input_pos = input_pos[:, curr_pos]
        curr_mask = causal_mask[:, curr_pos, None, :]

        h, logits = forward(
            model,
            tokens.clone(),
            input_pos=curr_input_pos,
            mask=curr_mask,
        )
        tokens = sample(logits[:, -1], temperature=temperature, top_k=top_k, top_p=top_p)
        # log.debug(f"{tokens.shape=}, {h.shape=}, {logits.shape=}")
        generated_tokens[:, curr_pos + 1] = tokens.squeeze()
        generated_logits[:, curr_pos] = logits[:, -1]
        generated_h[:, curr_pos] = h[:, -1]
        curr_pos += 1

    log.debug(f"{generated_tokens.shape=}, {generated_logits.shape=}, {stop_token_mask.shape=}")
    # mask out generated tokens in seqs that already hit a stop token
    if stop_tokens is not None:
        generated_tokens *= stop_token_mask
        generated_logits *= stop_token_mask[:, :, None]

    t_total = time.perf_counter() - t0

    # prevents saving a lot of info
    logits_topk = torch.topk(generated_logits, k=1024, dim=-1).values

    # limit storage usage
    return dict(
        tokens=generated_tokens[:, prompt_len:curr_pos].to(device="cpu", dtype=torch.int32),
        logits=logits_topk[:, prompt_len:curr_pos].to(device="cpu", dtype=torch.float16),
        hidden=generated_h[:, prompt_len:curr_pos].to(device="cpu", dtype=torch.float16),
        prompt_len=prompt_len,
        max_len=max_len,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        t_prefill=t_prefill,
        t_total=t_total,
    )
