# Self-consistency

## Download model weights

```bash
tune download meta-llama/Llama-3.2-1B-Instruct \
	--output-dir output/checkpoints/meta-llama/Llama-3.2-1B-Instruct \
	--ignore-patterns "original/consolidated.00.pth" \
	--hf-token $HF_TOKEN
```

## Run Eval

```bash
tune run eleuther_eval --config third_party/eleuther_eval/example_config.yaml \
	tasks=["truthfulqa_mc2","hellaswag"]
```
