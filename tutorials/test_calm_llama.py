
import torch

from transformers import (
    AutoTokenizer,
)
from llama2.model import CalmLlama2HF


@torch.no_grad()
def main():
    num_layers = 32
    head_type = "linear"
    checkpoint_path = (
        "output/train/07_03_2024_05_51_calm_linear_32_1024_None/best-checkpoint-6000.pt"
    )
    w = torch.load(checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained("output/weights/llama2-7b")
    model = CalmLlama2HF(num_calm_layers=num_layers, head_type=head_type)
    model.load_state_dict(w, strict=False)
    model = model.to("cuda")
    model.eval()

    prompt = "Hey, are you conscious? Can you talk to me?"
    inputs = tokenizer(prompt, return_tensors="pt")
    labels = tokenizer("I'm not conscious, but I can talk to you.", return_tensors="pt")
    print(inputs)
    generate_ids = model.generate(**inputs, max_length=30)
    generated_str = tokenizer.decode(generate_ids[0], skip_special_tokens=True)
    print(generated_str)


if __name__ == "__main__":
    main()
