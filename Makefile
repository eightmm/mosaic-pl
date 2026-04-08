PYTHON := uv run python

.PHONY: sync status mcp lint test template-db-build template-db-update

sync:
	uv sync --dev

status:
	uv run casp17-pl status

mcp:
	uv run casp17-pl-mcp

lint:
	uv run ruff check src

test:
	uv run pytest

template-db-build:
	bash scripts/build_template_search_dbs.sh \
		--sequence-fasta "$(SEQUENCE_FASTA)" \
		--structure-dir "$(STRUCTURE_DIR)" \
		--output-root "$(DB_ROOT)"

template-db-update:
	bash scripts/update_template_search_dbs.sh \
		--sequence-fasta "$(SEQUENCE_FASTA)" \
		--structure-dir "$(STRUCTURE_DIR)" \
		--output-root "$(DB_ROOT)"
