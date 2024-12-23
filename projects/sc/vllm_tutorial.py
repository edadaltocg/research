# GPU only
from vllm import LLM

llm = LLM(
    model="facebook/opt-125m",
    speculative_model="facebook/opt-125m",
    num_speculative_tokens=5,
)
outputs = llm.generate("The future of AI is")

for output in outputs:
    print(f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}")


llm = LLM("facebook/opt-13b", tensor_parallel_size=4)
output = llm.generate("San Franciso is a")
