from datasets import load_dataset

CACHE_DIR = "output/datasets"
config_list = [
    "Llama-3.2-1B-Instruct-evals__arc_challenge__details",
    "Llama-3.2-1B-Instruct-evals__bfcl_chat__details",
    "Llama-3.2-1B-Instruct-evals__gpqa__details",
    "Llama-3.2-1B-Instruct-evals__gsm8k__details",
    "Llama-3.2-1B-Instruct-evals__hellaswag_chat__details",
    "Llama-3.2-1B-Instruct-evals__ifeval__loose__details",
    "Llama-3.2-1B-Instruct-evals__ifeval__strict__details",
    "Llama-3.2-1B-Instruct-evals__infinite_bench__details",
    "Llama-3.2-1B-Instruct-evals__math__details",
    "Llama-3.2-1B-Instruct-evals__metrics",
    "Llama-3.2-1B-Instruct-evals__mgsm__details",
    "Llama-3.2-1B-Instruct-evals__mmlu__details",
    "Llama-3.2-1B-Instruct-evals__mmlu_french_chat__details",
    "Llama-3.2-1B-Instruct-evals__mmlu_german_chat__details",
    "Llama-3.2-1B-Instruct-evals__mmlu_hindi_chat__details",
    "Llama-3.2-1B-Instruct-evals__mmlu_italian_chat__details",
    "Llama-3.2-1B-Instruct-evals__mmlu_portugese_chat__details",
    "Llama-3.2-1B-Instruct-evals__mmlu_spanish_chat__details",
    "Llama-3.2-1B-Instruct-evals__mmlu_thai_chat__details",
    "Llama-3.2-1B-Instruct-evals__nexus__details",
    "Llama-3.2-1B-Instruct-evals__nih__multi_needle__details",
    "Llama-3.2-1B-Instruct-evals__openrewrite_chat__details",
]
for config in config_list:
    ds = load_dataset("meta-llama/Llama-3.2-1B-Instruct-evals", config, cache_dir=CACHE_DIR)

ds = load_dataset("openai/gsm8k", "main", cache_dir=CACHE_DIR)
