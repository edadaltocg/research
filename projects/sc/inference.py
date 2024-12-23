import os
from typing import Tuple

import torch
import torch.nn.functional as F
import torcheval.metrics.functional
import torchtune
import torchtune.models.llama3
import torchtune.models.llama3_2
from omegaconf import OmegaConf
from torch import Tensor
from torchtune import config, generation, training, utils
from torchtune.models import convert_weights
from torchtune.modules import TransformerDecoder

from research.utils.utils import safe_torch_load

logger = utils.get_logger("DEBUG")


def forward(
    self,
    tokens,
    *,
    mask=None,
    encoder_input=None,
    encoder_mask=None,
    input_pos=None,
) -> Tuple[Tensor, Tensor]:
    """
    Args:
        tokens (torch.Tensor): input tensor with shape ``[b x s]``
        mask (Optional[_MaskType]): Used to mask the scores after the query-key multiplication
            and before the softmax. This parameter is required during inference if caches have been setup.
            Either:

            A boolean tensor with shape ``[b x s x s]``, ``[b x s x self.encoder_max_cache_seq_len]``,
            or ``[b x s x self.encoder_max_cache_seq_len]`` if using KV-cacheing with encoder/decoder layers.
            A value of True in row ``i`` and column ``j`` means token ``i`` attends to token ``j``. A value of False means
            token ``i`` does not attend to token ``j``. If no mask is specified, a causal mask
            is used by default.

            A :class:`~torch.nn.attention.flex_attention.BlockMask` for document masking in a packed sequence
            created via `create_block_mask <https://pytorch.org/blog/flexattention/#mask-mods>`_. We  use
            :func:`~torch.nn.attention.flex_attention.flex_attention` when computing attention with block masks.
            Default is None.
        encoder_input (Optional[torch.Tensor]): Optional input embeds from the encoder. Shape ``[b x s_e x d_e]``
        encoder_mask (Optional[torch.Tensor]):  Boolean tensor defining a relational matrix between
            tokens and encoder embeddings. A True value at position ``i,j`` means token ``i`` can attend
            to embedding ``j`` in the decoder. Mask has shape ``[b x s x s_e]``. Default is None,
            but this is required during inference if the model has been setup with any layers
            which use encoder embeddings and caches have been setup.
        input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
            of each token. During training, this is used to indicate the positions
            of each token relative to its sample when packed, shape ``[b x s]``.
            During inference, this indicates the position of the current token.
            This parameter is required during inference if caches have been setup. Default is None.

    Returns:
        Union[torch.Tensor, List[torch.Tensor]]: output tensor with shape ``[b x s x v]`` or a list of layer
            output tensors defined by ``output_hidden_states`` with the
            final output tensor appended to the list.

    Note:
        At the very first step of inference, when the model is provided with a prompt,
        ``input_pos`` should contain the positions of all of the tokens in the prompt.
        For a single-batch prompt, or a batch of prompts with identical lengths, this
        will be ``torch.arange(prompt_length)``. For a batch of varying-length prompts,
        shorter prompts are left-padded and position ids are correspondingly right-shifted,
        thus positional ids should be of shape ``[b, padded_prompt_length]``.
        This is because we will need to retrieve the positional embeddings for each input id.
        In the subsequent steps, if the model has been setup with KV-caches, ``input_pos`` will contain
        the position(s) of the current token(s) ``torch.tensor([padded_prompt_length])``. Otherwise,
        ``input_pos`` will contain all the position ids up to the current token.

    Shape notation:
        - b: batch size
        - s: token sequence length
        - s_e: encoder sequence length
        - v: vocab size
        - d: token embed dim
        - d_e: encoder embed dim
        - m_s: max seq len
    """
    # input tensor of shape [b, s]
    seq_len = tokens.shape[1]

    self._validate_inputs(
        seq_len,
        mask=mask,
        encoder_input=encoder_input,
        encoder_mask=encoder_mask,
        input_pos=input_pos,
    )

    # shape: [b, s, d]
    h = self.tok_embeddings(tokens)

    hidden = []
    for i, layer in enumerate(self.layers):
        if i in self.output_hidden_states:
            hidden.append(h)
        # shape: [b, s, d]
        h = layer(
            h,
            mask=mask,
            encoder_input=encoder_input,
            encoder_mask=encoder_mask,
            input_pos=input_pos,
        )

    # shape: [b, s, d]
    h = self.norm(h)

    if self.num_output_chunks > 0:
        output = self.chunked_output(h)
    else:
        # shape: [b, seq_len, out_dim]
        output = self.output(h).float()

    # Output list if hidden states are requested, otherwise just the output
    output = output if not hidden else [*hidden, h, output]
    return h, output


def main():
    logger.info("Starting the inference process.")

    device = "cpu"
    model_dtype = torch.bfloat16
    logger.debug(f"Device set to {device}.")

    # Seed the random generator for reproducibility
    rng = torch.Generator(device=device).manual_seed(42)
    base_path = "output/checkpoints/meta-llama/Llama-3.2-1B-Instruct"

    # Loading the model
    logger.info("Initializing the llama3_2_1b model.")
    # model = torchtune.models.llama3_2.llama3_2_1b()
    cfg_model = OmegaConf.create({"_component_": "torchtune.models.llama3_2.llama3_2_1b"})
    with training.set_default_dtype(model_dtype):
        model: TransformerDecoder = config.instantiate(cfg_model)
    model.output_hidden_states = [15]

    # Load the state dict
    model_state_dict_path = os.path.join(base_path, "model.safetensors")
    logger.info(f"Loading model state dictionary from {model_state_dict_path}.")
    model_state_dict = safe_torch_load(model_state_dict_path, mmap=True)
    weights = convert_weights.hf_to_tune(
        model_state_dict,
        num_heads=32,
        num_kv_heads=8,
        dim=2048,
        head_dim=2048 // 32,
    )

    # Load the state dictionary into the model
    model.load_state_dict(weights)
    model = model.to(device=device, dtype=model_dtype)
    model.eval()
    logger.info("Model loaded and set to evaluation mode.")

    # Initializing the tokenizer
    logger.info("Initializing the tokenizer.")
    tokenizer = torchtune.models.llama3.llama3_tokenizer(
        os.path.join(base_path, "original", "tokenizer.model")
    )

    # Encode the prompt
    prompt_text = "Hi my name is"
    logger.info(f"Encoding the prompt: {prompt_text}")
    tokens = tokenizer.encode(prompt_text, add_eos=False, add_bos=False)

    # Input the prompt to the model
    prompt = torch.tensor(tokens, dtype=torch.int, device=device).unsqueeze(0)
    logger.info(f"{prompt=}")
    bsz, prompt_length = prompt.size()
    total_response_length = prompt_length + 1
    # Causal mask
    mask = torch.tril(
        torch.ones(
            size=(total_response_length, total_response_length),
            dtype=torch.bool,
            device=device,
        )
    ).unsqueeze(0)
    input_pos = torch.arange(total_response_length, device=device).unsqueeze(0)
    logger.info(f"{mask=}")
    logger.info(f"{input_pos=}")

    # Prefill step
    with torch.inference_mode():
        hidden, h, logits = forward(
            model,
            prompt,
            input_pos=input_pos[:, :prompt_length].squeeze(),
            mask=mask[:, :prompt_length, :prompt_length],
        )
    logger.info(f"{hidden.shape=}")
    logger.info(f"{h.shape=}")
    logger.info(f"{logits.shape=}")
    ppl = torcheval.metrics.functional.perplexity(logits, prompt)
    logger.info(f"{ppl=}")
    argmax_seq = logits.argmax(-1)
    ppl_amax = torcheval.metrics.functional.perplexity(logits, argmax_seq)
    logger.info(f"{ppl_amax=}")
    amax_tokens = tokenizer.decode(argmax_seq[0].tolist())
    logger.info(f"{amax_tokens=}")
    new_logits, o1, z = logits[:, -1], h[:, -1], hidden[:, -1]
    logger.info(f"{F.softmax(new_logits, dim=-1)=}")
    logger.info(f"{o1=}")
    logger.info(f"{z=}")
    ppl = torcheval.metrics.functional.perplexity(
        logits[:, -1].unsqueeze(0), argmax_seq[:, -1].unsqueeze(0)
    )
    logger.info(f"{ppl=}")

    return
    # Generate output
    logger.info("Starting generation.")
    top_k = None
    temperature = 1
    output, logits = generation.generate(
        model,
        prompt,
        max_generated_tokens=100,
        pad_id=0,
        temperature=temperature,
        top_k=top_k,
        rng=rng,
        stop_tokens=tokenizer.stop_tokens,
    )
    logger.info("Generation complete.")

    # Decode the output
    generated_text = tokenizer.decode(output[0].tolist())
    logger.info(f"Generated text: {generated_text}")
    print(generated_text)


if __name__ == "__main__":
    main()
