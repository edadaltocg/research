install:
	pipenv shell
	pipenv install --dev

test:
	python3 -m pytest -v -s .

fix:
	ruff check -v --fix .

watch:
	ruff check . --watch

format:
	ruff check --fix .
	ruff format .

static:
	mypy .

sync:
	rsync -avz --delete --exclude-from=.gitignore . jean-zay:/gpfswork/rech/oyx/umg45bq/research

live:
	fswatch -l 1 -o . | while read f; do make sync; done