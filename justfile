default:
    @just --list

# Initialize virtual environment and synchronize dependencies
install:
    uv venv
    uv sync --all-extras

# Format and fix code
fmt:
    uv run ruff format .
    uv run ruff check --fix --exit-zero .

# Lint code
lint:
    uv run ruff check .
    uv run mypy .

# Run tests
test:
    uv run pytest -v -s tests/

# Preview documentation locally
docs-preview:
    uv run mkdocs serve

# Build static documentation files
docs-build:
    uv run mkdocs build
