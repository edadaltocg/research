venv: ## Create virtual environment at .venv/
	uv venv

install: ## Install dependencies
	. .venv/bin/activate
	uv pip install torch torchvision torchaudio
	uv pip install -r pyproject.toml --all-extras
	uv pip install -e .

export: ## Export dependencies to .txt files
	uv export --all-extras > dev-requirements.txt
	uv export > requirements.txt

test: ## Run tests
	uv run pytest -v -s tests

fix: ## Fix code issues
	uv run ruff check -v --fix research

watch: ## Watch for code issues
	uv run ruff check research --watch

format: fix ## Format code
	uv run ruff format research

static: ## Static type checking
	uv run mypy research

docs: ## Build the documentation with MkDocs
	uv run mkdocs build

preview: ## Serve the documentation with MkDocs locally
	uv run mkdocs serve

help: ## Show this help message
	@uv run python -c "import re; \
	[[print(f'\033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"

.DEFAULT_GOAL := help
.PHONY: venv install export test fix watch format static docs preview help

