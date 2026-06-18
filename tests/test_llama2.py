from typing import Any

import pytest
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaModel


def convert_llama_state_dict(state_dict):
    return


@pytest.mark.skip(reason="Requires huge download and gated HuggingFace model access")
def test_llama():
    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    prompt = "Hey, are you conscious? Can you talk to me?"
    assert tokenizer is not None
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate
    model_any: Any = model
    tokenizer_any: Any = tokenizer
    generate_ids = model_any.generate(inputs.input_ids, max_length=30)
    tokenizer_any.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    LlamaModel.from_pretrained("meta-llama/Llama-2-7b-hf")
