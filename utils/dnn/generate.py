from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Sampler(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    @torch.no_grad()
    def forward(
        self,
        embedding: Tensor,
        hidden_states: Tensor,
        output_positions: Tensor,
        temperatures: Tensor,
        top_ps: Tensor,
        top_ks: Tensor,
        embedding_bias: Optional[Tensor] = None,
    ) -> Tensor:
        # Select the last element for each sequence.
        # (batch_size, input_len, hidden_size) -> (batch_size, hidden_size)
        hidden_states = hidden_states.index_select(1, output_positions).squeeze(dim=1)
        logits = torch.matmul(hidden_states, embedding.t())
        if embedding_bias is not None:
            logits += embedding_bias

        if temperatures is None:
            return torch.argmax(logits, dim=-1).squeeze(dim=-1)

        # Apply temperature scaling.
        logits.div_(temperatures.unsqueeze(dim=1))

        # Calculate probabilities with softmax.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)

        # Apply top-p, top-k.
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        top_ps_mask = (probs_sum - probs_sort) > top_ps.unsqueeze(dim=1)
        probs_sort = torch.where(top_ps_mask, 0, probs_sort)

        top_ks_mask = torch.arange(probs_idx.shape[-1], device=probs_idx.device)
        top_ks_mask = top_ks_mask.expand(probs_idx.shape[0], -1)
        top_ks_mask = top_ks_mask >= top_ks.unsqueeze(dim=1)
        probs_sort = torch.where(top_ks_mask, 0, probs_sort)

        # Re-normalization.
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        probs = torch.gather(probs_sort, dim=-1, index=torch.argsort(probs_idx, dim=-1))

        next_token_ids = torch.multinomial(
            probs, num_samples=1, replacement=True
        ).squeeze(dim=-1)
        return next_token_ids


@torch.no_grad()
def generate_naive(
    model, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None
):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    """
    model.eval()
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = (
            idx if idx.size(1) <= model.block_size else idx[:, -model.block_size :]
        )
        # forward the model to get the logits for the index in the sequence
        logits, _ = model(idx_cond)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("Inf")
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # either sample from the distribution or take the most likely element
        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def encode_tokens(tokenizer, string, bos=True, device="cuda"):
    tokens = tokenizer.encode(string)
    if bos:
        tokens = [tokenizer.bos_id()] + tokens
    return torch.tensor(tokens, dtype=torch.int, device=device).unsqueeze(0)


def multinomial_sample_one_no_sync(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[0, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def prefill(
    model, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs
) -> Tensor:
    # input_pos: [B, S]
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)[0]


def decode_n_tokens(model, max_new_tokens, input_toks, temperature=1.0, top_k=None):
    """Decode max_new_tokens tokens autoregressively (sequentially) from the model."""
    b, t = input_toks.size()
    output_tokens = torch.zeros(
        b, t + max_new_tokens, dtype=torch.int, device=input_toks.device
    )
    for _ in range(max_new_tokens):
        logits, _ = model(input_toks)
        idx_next, _ = sample(logits, temperature, top_k)
        yield idx_next


def beam_search():
    return


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens, temperature=1.0, top_k=None):
    device = next(model.parameters()).device
    tokenized_prompt = encode_tokens(tokenizer, prompt, bos=True, device=device)
    generated_tokens = torch.zeros(
        1, tokenized_prompt.size(1) + max_new_tokens, dtype=torch.int, device=device
    )
    # prefill the model with the prompt
    input_pos = torch.arange(tokenized_prompt.size(1), device=device)
    outputs = model(tokenized_prompt, input_pos)
    logits = outputs["logits"]
    idx_next, probs = sample(logits, temperature, top_k)
    # decode the rest of the tokens
    for idx in range(1, max_new_tokens):
        generated_tokens[0, idx] = idx_next
        outputs = model(generated_tokens[:, : idx + 1], input_pos[: idx + 1])
        logits = outputs["logits"]
        idx_next, probs = sample(logits, temperature, top_k)
    decoded_str = tokenizer.decode(generated_tokens[0].tolist())
    return decoded_str


@torch.no_grad()
def generate_speculative(model, draft):
    return
