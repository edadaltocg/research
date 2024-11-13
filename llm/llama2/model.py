from functools import partial
from dataclasses import dataclass
from pathlib import Path
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from typing import List, Optional, Tuple, Union
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
)
import torch
from transformers.utils import ModelOutput
from dnn.modeling.attention import GroupedQueryAttentionWithRoPEAndCache
from dnn.modeling.dense import LLaMAMLP
from dnn.modeling.transformer import (
    RMSNorm,
    TransformerWithRoPEAndKVCache,
    TransformerWithRoPEAndCacheLayer,
)
import logging


log = logging.getLogger(__name__)


class Llama2Train(TransformerWithRoPEAndKVCache):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        max_batch_size: int,
        hidden_size: int,
        max_seq_len: int,
        num_kv_heads: int,
        num_q_heads: int,
        num_layers: int,
    ):
        super().__init__(
            embedding=nn.Embedding(vocab_size, embed_dim),
            block_size=max_seq_len,
            max_batch_size=max_batch_size,
            encoder_layer=TransformerWithRoPEAndCacheLayer(
                mlp=LLaMAMLP(embed_dim, hidden_size),
                attn_mechanism=GroupedQueryAttentionWithRoPEAndCache(
                    embed_dim, num_kv_heads, num_q_heads
                ),
                norm_layer1=RMSNorm(768),
                norm_layer2=RMSNorm(768),
                norm_first=True,
            ),
            num_encoder_layers=num_layers,
            norm=RMSNorm(embed_dim),
            head=nn.Linear(embed_dim, vocab_size, bias=False),
        )


llama2_7B_config = dict(
    vocab_size=32000,
    embed_dim=4096,
    hidden_size=11008,
    num_layers=32,
    num_q_heads=32,
    max_seq_len=4096,
)

llama2_1B_config = dict(
    vocab_size=32000,
    embed_dim=1024,
    hidden_size=1024 * 4,
    num_layers=16,
    num_q_heads=16,
    max_seq_len=2048,
)
llama_1b_config = LlamaConfig(
    vocab_size=32000,
    hidden_size=2048,
    intermediate_size=2048 * 4,
    num_hidden_layers=16,
    num_attention_heads=8,
)


class LinearAdapter(nn.Module):
    def __init__(self, hidden_size: int = 4096) -> None:
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        torch.nn.init.zeros_(self.linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.linear(x))


class SiluAdapter(nn.Module):
    def __init__(self, hidden_size: int = 4096) -> None:
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear2 = nn.Linear(hidden_size, hidden_size, bias=False)
        torch.nn.init.zeros_(self.linear1.weight)
        torch.nn.init.zeros_(self.linear2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.linear2(F.silu(self.linear1(x))))


class LowRankAdapter(nn.Module):
    def __init__(self, hidden_size: int = 4096, rank: int = 16) -> None:
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.linear1 = nn.Linear(hidden_size, rank, bias=False)
        self.linear2 = nn.Linear(rank, hidden_size, bias=False)
        torch.nn.init.zeros_(self.linear1.weight)
        torch.nn.init.zeros_(self.linear2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.linear2(F.silu(self.linear1(x))))


@dataclass
class CausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    intermediate_logits: Optional[List[torch.FloatTensor]] = None
    intermediate_losses: Optional[List[torch.FloatTensor]] = None


class CalmLlama2HF(LlamaForCausalLM):
    def __init__(
        self,
        config: LlamaConfig = LlamaConfig(),
        num_calm_layers: int = 8,
        head_type: str = "linear",
    ):
        super().__init__(config)
        self.num_calm_layers = num_calm_layers
        self.head_type = head_type
        self.head_cls = {
            "linear": LinearAdapter,
            "silu": SiluAdapter,
            "lowrank16": partial(LowRankAdapter, rank=16),
            "lowrank32": partial(LowRankAdapter, rank=32),
        }[head_type]

        self.reset_calm_layers()

    def reset_calm_layers(self):
        self.calm_layers = nn.ModuleList(
            [
                self.head_cls(self.config.hidden_size)
                for _ in range(self.num_calm_layers)
            ]
        )
        equally_spaced = max(1, self.config.num_hidden_layers // self.num_calm_layers)
        self.calm_indexes = {i: equally_spaced * i for i in range(self.num_calm_layers)}
        # e.g., 32, 8 -> {0: 0, 1: 4, 2: 8, 3: 12, 4: 16, 5: 20, 6: 24, 7: 28}

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )

            log.debug(f"{outputs.keys()=}")
            hidden_states = outputs[0]
            all_hidden_states = outputs["hidden_states"]
            # output_hidden_states = outputs[1]
            log.debug(f"{hidden_states.shape=}")
            if self.config.pretraining_tp > 1:
                lm_head_slices = self.lm_head.weight.split(
                    self.vocab_size // self.config.pretraining_tp, dim=0
                )
                logits = [
                    F.linear(hidden_states, lm_head_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
                logits = torch.cat(logits, dim=-1)
            else:
                logits = self.lm_head(hidden_states)
        logits = logits
        last_head_logits = logits
        log.debug(f"{logits.shape=}")
        consistency_labels = logits.argmax(dim=-1)
        log.debug(f"{consistency_labels.shape=}")

        loss = None
        # additional_logits = []
        # additional_losses = []

        # calm loss
        loss_fct = CrossEntropyLoss()
        # shift_logits = logits[..., :-1, :].contiguous()
        # shift_labels = input_ids[..., 1:].contiguous()
        # shift_logits = shift_logits.view(-1, self.config.vocab_size)
        # shift_labels = shift_labels.view(-1)
        # loss = loss_fct(shift_logits, shift_labels)
        loss = torch.zeros(1, device=logits.device)
        w_den = sum(self.calm_indexes.values())
        for layer_idx, hidden_state_idx in self.calm_indexes.items():
            hidden_state = all_hidden_states[hidden_state_idx]
            log.debug(f"{hidden_state.shape=}")
            calm_hidden_state = self.calm_layers[layer_idx](hidden_state)
            log.debug(f"{calm_hidden_state.shape=}")
            logits = self.lm_head(calm_hidden_state).float()

            # additional_logits.append(logits)

            log.debug(f"{logits.shape=}")
            shift_logits = logits[..., :-1, :].contiguous()
            log.debug(f"{shift_logits.shape=}")
            shift_labels = consistency_labels[..., 1:].contiguous().clone()
            log.debug(f"{shift_labels.shape=}")
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            log.debug(f"{shift_logits.shape=}")
            shift_labels = shift_labels.view(-1)
            log.debug(f"{shift_labels.shape=}")
            calm_loss = loss_fct(shift_logits, shift_labels)
            w_i = hidden_state_idx / w_den
            # additional_losses.append(calm_loss)

            loss += calm_loss * (w_i)

        # if labels is not None:
        #     # Shift so that tokens < n predict n
        #     shift_logits = logits[..., :-1, :].contiguous()
        #     shift_labels = labels[..., 1:].contiguous()
        #     # Flatten the tokens
        #     loss_fct = CrossEntropyLoss()
        #     shift_logits = shift_logits.view(-1, self.config.vocab_size)
        #     shift_labels = shift_labels.view(-1)
        #     # Enable model parallelism
        #     shift_labels = shift_labels.to(shift_logits.device)
        #     loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=last_head_logits,
            # past_key_values=outputs.past_key_values,
            # hidden_states=outputs.hidden_states,
            # attentions=outputs.attentions,
            # intermediate_logits=additional_logits,
            # intermediate_losses=additional_losses,
        )


class CalmLlama2HFNonLinear(CalmLlama2HF):
    def reset_calm_layers(self):
        self.calm_layers = nn.ModuleList(
            [SiluAdapter(self.config.hidden_size) for _ in range(self.num_calm_layers)]
        )
        equally_spaces = self.config.num_hidden_layers // (
            self.num_calm_layers + 1
        )  # floor
        self.calm_indexes = {
            i: equally_spaces * (i + 1) for i in range(self.num_calm_layers)
        }


def save_pretrained_llama(save_dir: str = "output/weights/llama2-7b"):
    save_root = Path(save_dir)
    save_root.mkdir(parents=True, exist_ok=True)
    model1 = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    model1_state_dict = model1.state_dict()
    torch.save(model1_state_dict, save_root / "llama2-7b-hf.pth")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.save_pretrained(save_root)


def prepare_calm_llama(
    save_dir: str = "output/weights/llama2-7b",
    device: str = "cuda:0",
    num_layers: int = 8,
    head_type: str = "linear",
):
    save_root = Path(save_dir)
    tokenizer = AutoTokenizer.from_pretrained(save_root)
    model = CalmLlama2HF(num_calm_layers=num_layers, head_type=head_type)
    state_dict = torch.load(save_root / "llama2-7b-hf.pth")
    model.load_state_dict(state_dict, strict=False)

    print(model.dtype)
    print("model loaded")

    model.model.requires_grad_(False)
    model.lm_head.requires_grad_(False)
    model.calm_layers.requires_grad_(True)

    print(model)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.size(), param.requires_grad, param.dtype, param.device)
    return model, tokenizer


def quick_calm_llama_test():
    model, tokenizer = prepare_calm_llama()
    prompt = "Hey, are you conscious? Can you talk to me?"
    inputs = tokenizer(prompt, return_tensors="pt")
    labels = tokenizer("I'm not conscious, but I can talk to you.", return_tensors="pt")
    print(inputs)
    generate_ids = model.generate(**inputs, max_length=30)
    print(
        tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
    )
    # "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."


def debug():
    torch.set_default_device("meta")
    model = CalmLlama2HF()
    for name, param in model.named_buffers():
        print(name, param.size(), param.requires_grad, param.dtype, param.device)
    state_dict = torch.load("output/weights/llama2-7b-hf.pth")
    for name, param in state_dict.items():
        print(name, param.size(), param.requires_grad, param.dtype, param.device)


hyperparameters = dict(
    adapter_type=["linear", "silu"],
    calm_layers=[16, 32],
    rank=["full", 16, 32, 256, 512, 1024, 2048],
    model_types=["llama-7b", "mistral-7b"],
)
# (8 calm layers) : 1000 * 2048 * 12 * 2 = 60M tok/h
# 2T tokens -> 1% = 20B tokens -> would take 333h

if __name__ == "__main__":
    import fire

    fire.Fire(
        {
            "save_pretrained_llama": save_pretrained_llama,
            "test": quick_calm_llama_test,
            "debug": debug,
        }
    )
