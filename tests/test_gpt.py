from typing import OrderedDict

import pysnooper
import torch
from transformers import GPT2LMHeadModel, GPT2Model, GPT2Tokenizer

from gpt.model import GPT


def convert_gpt2_state_dict(state_dict: OrderedDict) -> OrderedDict:
    # HF
    # 'transformer.wte.weight', 'transformer.wpe.weight', 'transformer.h.0.ln_1.weight', 'transformer.h.0.ln_1.bias', 'transformer.h.11.ln_1.weight', 'transformer.h.11.ln_1.bias', 'transformer.h.11.attn.c_attn.weight', 'transformer.h.11.attn.c_attn.bias', 'transformer.h.11.attn.c_proj.weight', 'transformer.h.11.attn.c_proj.bias', 'transformer.h.11.ln_2.weight', 'transformer.h.11.ln_2.bias', 'transformer.h.11.mlp.c_fc.weight', 'transformer.h.11.mlp.c_fc.bias', 'transformer.h.11.mlp.c_proj.weight', 'transformer.h.11.mlp.c_proj.bias', 'transformer.ln_f.weight', 'transformer.ln_f.bias', 'lm_head.weight'
    # This
    # 'encoder.embedding.weight', 'encoder.pos_encoding.pe.weight', 'encoder.layers.0.attn.qkv_proj.weight', 'encoder.layers.11.attn.qkv_proj.weight', 'encoder.layers.11.attn.qkv_proj.bias', 'encoder.layers.11.attn.out_proj.weight', 'encoder.layers.11.attn.out_proj.bias', 'encoder.layers.11.mlp.linear1.weight', 'encoder.layers.11.mlp.linear1.bias', 'encoder.layers.11.mlp.linear2.weight', 'encoder.layers.11.mlp.linear2.bias', 'encoder.layers.11.norm1.weight', 'encoder.layers.11.norm1.bias', 'encoder.layers.11.norm2.weight', 'encoder.layers.11.norm2.bias', 'encoder.norm.weight', 'encoder.norm.bias', 'head.weight'
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = (
            k.replace("transformer.", "")
            .replace("h.", "layers.")
            .replace("attn.c_attn", "attn.qkv_proj")
            .replace("attn.c_proj", "attn.out_proj")
            .replace("mlp.c_fc", "mlp.linear1")
            .replace("mlp.c_proj", "mlp.linear2")
            .replace("ln_1", "norm1")
            .replace("ln_2", "norm2")
            .replace("ln_f", "norm")
            .replace("wte", "embedding")
            .replace("wpe", "pos_encoding.pe")
            .replace("ln_f", "norm")
            .replace("lm_head", "head")
        )
        # E               size mismatch for encoder.layers.11.attn.qkv_proj.weight: copying a param with shape torch.Size([768, 2304]) from checkpoint, the shape in current model is torch.Size([2304, 768]).
        # E               size mismatch for encoder.layers.11.mlp.linear1.weight: copying a param with shape torch.Size([768, 3072]) from checkpoint, the shape in current model is torch.Size([3072, 768]).
        # E               size mismatch for encoder.layers.11.mlp.linear2.weight: copying a param with shape torch.Size([3072, 768]) from checkpoint, the shape in current model is torch.Size([768, 3072]).
        if (
            "attn.qkv_proj" in new_key
            or "mlp.linear1" in new_key
            or "mlp.linear2" in new_key
        ):
            v = v.t()
        new_state_dict[new_key] = v
    return new_state_dict


@pysnooper.snoop()
def test_gpt():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2")
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors="pt")
    # 'input_ids', 'attention_mask'
    with torch.no_grad():
        output = model(**encoded_input)
    # odict_keys(['last_hidden_state', 'past_key_values'])
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    print(model)

    with torch.no_grad():
        output = model(**encoded_input)
    # odict_keys(['logits', 'past_key_values'])
    logits = output.logits

    # load gpt
    gpt = GPT(
        vocab_size=50257, embed_dim=768, max_seq_len=1024, num_heads=12, num_layers=12
    )
    print(gpt)
    pretrained_w = model.state_dict()
    new_w = convert_gpt2_state_dict(pretrained_w)
    print(set(new_w.keys()) - set(gpt.state_dict().keys()))
    incompatible_keys = gpt.load_state_dict(new_w, strict=False)
    print(incompatible_keys)

    x = encoded_input.input_ids
    with torch.no_grad():
        output = gpt(x)

    print((logits - output).abs().max())
