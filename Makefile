default: check

help: ## Show make commands.
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

VIRTUAL_ENV ?= .venv

ifeq ($(OS),WINDOWS_NT)
	BIN=$(VIRTUAL_ENV)/Scripts
else
	BIN=$(VIRTUAL_ENV)/bin
endif

PYTHON=$(BIN)/python

.PHONY: uv-installed
uv-installed:
	@$(eval UV:=$(shell command -v uv))

.PHONY: info
info: uv-installed  ## Print out some variables to check your environment.
	@echo Using venv $(VIRTUAL_ENV)
	@echo Using python from $(PYTHON)
	@echo Using uv from $(UV)

.PHONY: lock-check
lock-check: uv-installed  ## Confirm uv.lock matches project requirements.
	$(UV) lock --check

.PHONY: sync
sync: uv-installed  ## Install package and all extras with uv.
	$(UV) sync --all-extras
	@echo All dependencies installed to your virtual env. Pick a name for the ipython kernel if you want to run notebooks:
	@echo   python -m ipykernel install --user --name=finm37000 --display-name=\"FINM 37000 venv\"

.PHONY: lint
lint:  ## Checks if the source currently matches code conventions using ruff.
	$(PYTHON) -m ruff check

.PHONY: lint-fix
lint-fix:  ## Checks if everything matches code conventions and fixes those which are trivial to do so using ruff.
	$(PYTHON) -m ruff check --fix

.PHONY: format-check
format-check:  ## Check that source files are formatted.
	$(PYTHON) -m ruff format --check

.PHONY: format
format:  ## Format source files.
	$(PYTHON) -m ruff format

.PHONY: typecheck
typecheck:  ## Typecheck with mypy.
	$(PYTHON) -m mypy

.PHONY: test
test:  ## Runs the tests.
	$(PYTHON) -m pytest
	@echo Tests pass

.PHONY: check
check: info lock-check format-check lint typecheck test  ## Runs all checks without file or environment modifications.

.PHONY: fix ## Runs all checks with fixes.
fix: info sync format lint-fix typecheck test

.PHONY: clean
clean:  ## Cleans up everything.
	rm -rf .venv .venv-* dist *.egg-info .pytest_cache .mypy_cache .ruff_cache

PYTHON_VERSIONS := 3.12 3.13 3.14
TEST_PY_TARGETS := $(addprefix test-py,$(PYTHON_VERSIONS))

.PHONY: test-all
test-all: $(TEST_PY_TARGETS)  ## Run tests against Python versions 3.12-3.14. Consider make -j3 test-all for parallel builds.

test-py%:
	UV_PROJECT_ENVIRONMENT=.venv-$* uv run --python $* pytest

