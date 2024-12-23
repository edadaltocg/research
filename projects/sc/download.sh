#!/usr/bin/env bash

# Define the array of model IDs
ids=(
	meta-llama/Llama-3.2-1B-Instruct
	meta-llama/Llama-3.2-3B-Instruct
	meta-llama/Llama-3.1-8B-Instruct
	meta-llama/Llama-3.3-70B-Instruct
	google/gemma-2b
	google/gemma-7b
	mistralai/Mistral-7B-v0.1
	mistralai/Ministral-8B-Instruct-2410
	Qwen/Qwen2.5-1.5B-Instruct
	Qwen/Qwen2.5-3B-Instruct
	Qwen/Qwen2.5-7B-Instruct
	Qwen/Qwen2.5-14B-Instruct
	Qwen/Qwen2.5-32B-Instruct
	Qwen/Qwen2.5-72B-Instruct
)

# Configuration variables
OUTPUT_DIR_PREFIX="output/checkpoints"
IGNORE_PATTERN="original/consolidated**pth,*.gguf"

# Check if required environment variable is set
if [[ -z "$HF_TOKEN" ]]; then
  echo "Error: HF_TOKEN is not set."
  exit 1
fi

# Dry run option
DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
  DRY_RUN=true
fi

for id in "${ids[@]}"; do
  output_dir="$OUTPUT_DIR_PREFIX/$id"

  echo "Downloading $id to $output_dir"

  if [ "$DRY_RUN" = false ]; then
    # Attempt to download the model
    if ! .venv/bin/tune download "$id" \
      --output-dir "$output_dir" \
      --ignore-patterns "$IGNORE_PATTERN" \
      --hf-token "$HF_TOKEN"; then
      echo "Failed to download $id"
      continue
    fi

    echo "Successfully downloaded $id"
  else
    echo "[Dry Run] Would invoke: tune download $id --output-dir $output_dir --ignore-patterns $IGNORE_PATTERN --hf-token *****"
  fi

done

echo "All downloads attempted."
