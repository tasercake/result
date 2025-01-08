# phony trick from https://keleshev.com/my-book-writing-setup/
.PHONY: phony

install: phony
	uv sync

lint: phony
	uv run ruff check .

type-check: phony
	uv run mypy

test: phony
	uv run pytest

docs: phony
	uv run --python 3.10 lazydocs \
		--overview-file README.md \
		--src-base-url https://github.com/rustedpy/result/blob/main/ \
		./src/result
