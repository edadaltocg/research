from transformers import AutoTokenizer, LlamaForCausalLM, LlamaModel


def convert_llama_state_dict(state_dict):
    return


def test_llama():
    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    prompt = "Hey, are you conscious? Can you talk to me?"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate
    generate_ids = model.generate(inputs.input_ids, max_length=30)
    tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    my_model = LlamaModel.from_pretrained("meta-llama/Llama-2-7b-hf")
