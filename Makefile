.PHONY: lint typecheck test check format

lint:
	uv run ruff check .

format:
	uv run ruff format .
	uv run ruff check . --fix

typecheck:
	uv run mypy src tests

test:
	uv run pytest --cov=src --cov-report=term-missing

check: format typecheck test
