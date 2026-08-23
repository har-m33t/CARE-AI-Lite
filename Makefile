.PHONY: help install check lint type test test-security test-db db-up db-check pin-models eval-smoke reproduce clean

VENV := .venv
PY   := $(VENV)/bin/python
UV   := $(HOME)/.local/bin/uv

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Sync the virtualenv from pyproject
	$(UV) sync --extra dev

check: lint type test ## Lint + typecheck + unit tests (no model, no DB)

lint: ## Ruff
	$(VENV)/bin/ruff check carelite tests
	$(VENV)/bin/ruff format --check carelite tests

type: ## Mypy
	$(VENV)/bin/mypy carelite

test: ## Unit tests (excludes inference and db markers)
	$(VENV)/bin/pytest -m "not inference and not db" -q

test-security: ## Adversarial input corpus: injection, PHI, red-flag
	$(VENV)/bin/pytest -m security -q

test-db: ## Tests needing a live Postgres
	$(VENV)/bin/pytest -m db -q

db-up: ## Apply the schema to an existing database (see REPRODUCE.md to create it)
	$(PY) -c "from carelite.db import apply_schema; apply_schema(); print('schema applied')"

db-check: ## Wave-0 gate: extension, tables, three-way join
	$(PY) -m carelite.db.check

pin-models: ## Record Ollama digests for every configured model tag
	$(PY) -m carelite.models.pin

eval-smoke: ## 5 scenarios x all conditions, end to end
	$(PY) -m carelite.eval.smoke

reproduce: ## Cold rebuild of every figure and table from the database
	$(PY) -m carelite.repro

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__
