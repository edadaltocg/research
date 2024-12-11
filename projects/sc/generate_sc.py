import logging
import time
import itertools
from typing import Any, Dict, Union

import torch
from torch import Tensor
from torchtune.modules import TransformerDecoder
from torchtune.modules.tokenizers import TikTokenBaseTokenizer


from projects.sc.inference import forward
from research.utils.logging import setup_logger
from research.llm.sample import sample

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
    tokenizer: TikTokenBaseTokenizer,
    prompt: str,
    *,
    n_paths=5,
    max_new_tokens=1,
    temperature=1,
    top_p=None,
    top_k=None,
    stop_tokens=None,
) -> Dict[str, Union[Tensor, Any]]:
    """The main entry point for generating tokens from a prompt."""

    bsz = n_paths
    if not stop_tokens:
        stop_tokens = [tokenizer.eos_id]
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    tokens = tokenizer.encode(prompt, add_eos=False, add_bos=False)
    model_inputs = torch.tensor(tokens, device=device).unsqueeze(0).expand(bsz, -1)
    prompt_len = model_inputs.size(1)
    total_response_length = prompt_len + max_new_tokens

    # Setup KV cache
    with device:
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
    )
    input_pos = torch.arange(total_response_length)

    # Prefill step
    generated_tokens = model_inputs.clone()
    t0 = time.perf_counter()
    h, logits = forward(
        model,
        model_inputs,
        mask=causal_mask[None, :prompt_len],
        input_pos=input_pos[None, :prompt_len],
    )
    tokens = sample(logits[:, -1], temperature=temperature, top_k=top_k)
    t_prefill = time.perf_counter() - t0

    generated_tokens = torch.cat([generated_tokens, tokens], dim=-1)
    generated_logits = logits.clone()
    generated_h = h.clone()
    curr_pos = prompt_len

    # Stop token
    stop_token_reached = torch.zeros(bsz, dtype=torch.bool, device=device)
    stop_tokens = torch.tensor(stop_tokens, device=tokens.device, dtype=tokens.dtype)
    # everything in stop_token_mask starts as 1s, and we'll set them to 0 for sequences
    # that already hit a stop token
    stop_token_mask = torch.ones((bsz, prompt_len + 1), dtype=torch.int32, device=tokens.device)

    stop_token_reached = _update_stop_tokens_tracker(tokens, stop_tokens, stop_token_reached)
    if stop_token_reached.all().item():
        return generated_tokens, generated_logits

    # Continue generating
    for _ in range(max_new_tokens - 1):
        # Update stop tokens mask
        stop_token_mask = torch.cat([stop_token_mask, ~stop_token_reached.reshape(bsz, 1)], dim=-1)
        # Incremental decoding
        curr_input_pos = input_pos[None, curr_pos]
        curr_mask = causal_mask[None, curr_pos, None, :]

        h, logits = forward(
            model,
            tokens.clone(),
            input_pos=curr_input_pos,
            mask=curr_mask,
        )
        tokens = sample(logits[:, -1], temperature=temperature, top_k=top_k)
        generated_tokens = torch.cat([generated_tokens, tokens], dim=-1)
        generated_logits = torch.cat([generated_logits, logits], dim=1)
        generated_h = torch.cat([generated_h, h], dim=1)
        curr_pos += 1

        stop_token_reached = _update_stop_tokens_tracker(tokens, stop_tokens, stop_token_reached)
        if stop_token_reached.all().item():
            break

    # mask out generated tokens in seqs that already hit a stop token
    if stop_tokens is not None:
        generated_tokens *= stop_token_mask
        generated_logits *= stop_token_mask[:, :-1, None]

    t = time.perf_counter() - t0

    # Translate tokens back to text
    answers = []
    for i, gt in enumerate(generated_tokens):
        decoded = tokenizer.decode(gt.tolist()[prompt_len:])
        log.info(f"{i}: {decoded}")
        answers.append(decoded)

    # Log metrics
    tokens_per_second = len(generated_tokens) / t
    _log_metrics(model, total_time=t, prefill_time=t_prefill, tokens_per_second=tokens_per_second)

    return dict(
        all_tokens=generated_tokens,
        all_logits=generated_logits,
        all_h=generated_h,
        tokens=generated_tokens[:, prompt_len:],
        logits=generated_logits[:, prompt_len:],
        h=generated_h[:, prompt_len:],
        answers=answers,
    )
