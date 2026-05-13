PYTHON := uv run python

.PHONY: sync status mcp lint lint-lg test template-db-build template-db-update

sync:
	uv sync --dev

status:
	uv run casp17-pl status

mcp:
	uv run casp17-pl-mcp

lint:
	uv run ruff check src

# Validate every LG-format file under experiments/submissions against the
# CASP17 LG grammar (docs/casp17_lg_format.md). Exits non-zero on any ERROR.
# Override with: make lint-lg LG_FILES="path/to/foo.lg"
LG_FILES ?= $(wildcard experiments/submissions/*.lg)
lint-lg:
	@if [ -z "$(LG_FILES)" ]; then \
		echo "no LG files found under experiments/submissions/"; exit 0; \
	fi
	uv run python scripts/lint_lg_submission.py $(LG_FILES)

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
