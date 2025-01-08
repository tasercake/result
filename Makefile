# phony trick from https://keleshev.com/my-book-writing-setup/
.PHONY: phony

install: phony
	uv sync

lint: phony lint-flake lint-mypy

lint-flake: phony
	uv run ruff check .

lint-mypy: phony
	uv run mypy

test: phony
	uv run pytest

docs: phony
	uv run lazydocs \
		--overview-file README.md \
		--src-base-url https://github.com/rustedpy/result/blob/main/ \
		./src/result
