.DEFAULT_GOAL := help
PKG := gipacount
CONFIG ?= configs/smoke.yaml

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'

.PHONY: sync
sync: ## Install runtime + dev dependencies
	uv sync

.PHONY: test
test: ## Run the test suite
	uv run pytest

.PHONY: test-fast
test-fast: ## Run the test suite, skipping slow/gpu/network tests
	uv run pytest -m "not slow and not gpu and not network"

.PHONY: cov
cov: ## Run tests with coverage
	uv run pytest --cov --cov-report=term-missing

.PHONY: lint
lint: ## Lint and format-check
	uv run ruff check .
	uv run ruff format --check .

.PHONY: fmt
fmt: ## Auto-format and auto-fix
	uv run ruff check --fix .
	uv run ruff format .

.PHONY: typecheck
typecheck: ## Static type check
	uv run mypy src

.PHONY: check
check: lint typecheck test-fast ## Everything CI runs

.PHONY: smoke
smoke: ## Tiny end-to-end train+eval on the synthetic dataset (CPU, <60s)
	uv run $(PKG) train --config configs/smoke.yaml

.PHONY: data
data: ## Prepare / verify datasets (see scripts/download_data.py --help)
	uv run python scripts/download_data.py --help

.PHONY: train
train: ## Train with CONFIG=configs/xxx.yaml
	uv run $(PKG) train --config $(CONFIG)

.PHONY: eval
eval: ## Evaluate a checkpoint: make eval CKPT=checkpoints/best.pt
	uv run $(PKG) eval --checkpoint $(CKPT)

.PHONY: clean
clean: ## Remove caches and build artefacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
