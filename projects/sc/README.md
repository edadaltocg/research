# Self-consistency

## Download model weights

```bash
ids=(
	meta-llama/Llama-3.2-1B-Instruct
	meta-llama/Llama-3.2-3B-Instruct
	meta-llama/Llama-3.1-8B-Instruct
	meta-llama/Llama-3.3-70B-Instruct
)

for id in "${ids[@]}"; do
	tune download "$id" \
		--output-dir "output/checkpoints/$id" \
		--ignore-patterns "original/consolidated.00.pth" \
		--hf-token $HF_TOKEN
done
```

## Run Eval

```bash
tune run eleuther_eval --config third_party/eleuther_eval/example_config.yaml \
	tasks=["truthfulqa_mc2","hellaswag"]
```
