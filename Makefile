PYTHON := $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; elif command -v python3 >/dev/null 2>&1; then echo python3; else echo python; fi)
PYTEST := $(PYTHON) -m pytest
PYTEST_PRETTY_OPTS := -o addopts="--strict-markers --color=yes"
FAST_MARKERS := not integration and not adapter and not slow

.PHONY: help test test-fast test-all

help:
	@printf "Targets:\n"
	@printf "  make test       Run core test subset with pretty output\n"
	@printf "  make test-fast  Run core subset and stop on first failure\n"
	@printf "  make test-all   Run full test suite\n"

test:
	$(PYTEST) $(PYTEST_PRETTY_OPTS) -m "$(FAST_MARKERS)"

test-fast:
	$(PYTEST) $(PYTEST_PRETTY_OPTS) -m "$(FAST_MARKERS)" -x --maxfail=1

test-all:
	$(PYTEST) $(PYTEST_PRETTY_OPTS) tests
