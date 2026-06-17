default:
    @just --list

# Initialize virtual environment and synchronize dependencies
install:
    uv sync --frozen --all-extras

# Format and fix code
fmt:
    uv run ruff format .
    uv run ruff check --fix --exit-zero .

# Lint code
lint:
    uv run ruff check .
    uv run ty check --exit-zero .

# Run tests
test:
    uv run pytest -v -s tests/

# Preview documentation locally
docs-preview:
    uv run scripts/gen_ref_pages.py
    uv run properdocs serve

# Build static documentation files
docs-build:
    uv run scripts/gen_ref_pages.py
    uv run properdocs build
