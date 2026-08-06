.DEFAULT_GOAL := help
.PHONY: help install fmt lint type check test test-cov integration vcr-replay vcr-record \
        doctor stack-up stack-down stack-logs eval refresh-cassettes sync-langfuse clean

UV ?= uv run
PY_DIRS := $(wildcard src tests examples scripts)

help: ## Show this help message
	@awk 'BEGIN {FS = ":.*?## "; printf "Usage: make \033[36m<target>\033[0m\n\nTargets:\n"} \
	/^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Sync the uv environment (creates .venv if needed)
	uv sync

fmt: ## Format source with ruff
	$(UV) ruff format $(PY_DIRS)

lint: ## Lint source with ruff
	$(UV) ruff check $(PY_DIRS)

type: ## Type-check with pyright (strict on src/strata_forge)
	$(UV) pyright

check: lint type ## Run lint + type-check

test: ## Run unit tests
	$(UV) pytest tests/unit

test-cov: ## Run unit tests with coverage report
	$(UV) pytest tests/unit --cov=strata_forge --cov-report=term-missing

integration: ## Run integration tests (requires `make stack-up` first)
	$(UV) pytest -m integration

vcr-replay: ## Replay committed VCR cassettes (no live keys needed)
	$(UV) pytest tests/vcr

vcr-record: ## Record fresh VCR cassettes (requires live keys, RECORD=1)
	RECORD=1 $(UV) pytest tests/vcr

doctor: ## Run the `forge doctor` diagnostic command
	$(UV) forge doctor

stack-up: ## Start the local dev stack (Langfuse + Postgres + Qdrant + Redis)
	docker compose -f docker/compose.yaml up -d

stack-down: ## Stop the local dev stack
	docker compose -f docker/compose.yaml down

stack-logs: ## Tail logs from the local dev stack
	docker compose -f docker/compose.yaml logs -f

eval: ## Run the evaluation suite (Phase 2.4)
	@echo "[forge] eval suite not yet implemented — lands in Phase 2.4"

refresh-cassettes: vcr-record ## Re-record every VCR cassette (alias for `vcr-record`)

sync-langfuse: ## Sync prompts/datasets to Langfuse (Phase 2)
	@echo "[forge] Langfuse sync not yet implemented — lands in Phase 2"

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .pyright .mypy_cache .hypothesis \
	       .coverage htmlcov build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
