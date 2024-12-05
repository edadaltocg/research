install:
	uv pip install -r pyproject.toml --all-extras
	uv pip install -e .

test:
	uv run pytest -v -s tests

fix:
	uv run ruff check -v --fix research

watch:
	uv run ruff check research --watch

format: fix
	uv run ruff format research

static:
	uv run mypy research

