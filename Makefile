PYTHON := $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; elif command -v python3 >/dev/null 2>&1; then echo python3; else echo python; fi)
PYTEST := $(PYTHON) -m pytest
MYPY := $(PYTHON) -m mypy
RUFF := $(PYTHON) -m ruff
PYTEST_PRETTY_OPTS := -o addopts="--strict-markers --color=yes"
PYTEST_EXTRA_OPTS ?=

MODULE_config := tests/config/test_parser.py tests/config/test_validators.py
MODULE_adapters := tests/adapters/test_cli_adapter.py tests/adapters/test_compat_wrappers.py tests/adapters/test_gui_simulation_adapter.py
MODULE_app_bridge := tests/test_engine_and_gui_bridge.py tests/test_service_parity.py
MODULE_pipeline := tests/integration/test_pipeline_engine.py tests/test_pipeline_engine_errors.py
MODULE_diffusion := tests/test_1d_limit.py tests/test_mass_conservation.py tests/test_symmetry.py tests/test_cap_models.py
MODULE_oxidation := tests/test_oxidation_fractional_mask.py tests/test_oxidation_p2.py tests/test_deck_cap_model.py
MODULE_metrics := tests/test_metrics.py tests/test_iso_area_methods.py tests/test_analyze_iso_area_option.py
MODULE_export := tests/test_export_manager.py tests/test_vtk_writer.py tests/test_history.py
MODULE_quality := tests/test_marker_policy.py tests/test_selfcheck.py

INTEGRATION_pipeline := tests/integration/test_pipeline_engine.py tests/test_analyze_iso_area_option.py
INTEGRATION_oxidation := tests/test_oxidation_p2.py tests/test_deck_cap_model.py
INTEGRATION_gui := tests/test_engine_and_gui_bridge.py tests/test_service_parity.py
INTEGRATION_history := tests/test_history.py

.PHONY: help help-tests test test-all test-fast \
	test-module_config test-module_adapters test-module_app_bridge test-module_pipeline \
	test-module_diffusion test-module_oxidation test-module_metrics test-module_export test-module_quality \
	test-integration_all test-integration_pipeline test-integration_oxidation test-integration_gui test-integration_history \
	test-one test-k typecheck lint test-cov check refactor-check

help:
	@printf "프로젝트 테스트 타깃 (GNU make 기본 도움말은 'make --help')\n"
	@printf "  make test                      전체 테스트 실행 (최종 게이트)\n"
	@printf "  make test-fast                 전체 테스트 fail-fast 실행\n"
	@printf "  make test-module_<기능>        기능(모듈) 단위 테스트 실행\n"
	@printf "  make test-integration_<기능>   통합 테스트 실행\n"
	@printf "  make test-one TEST=...         단일 테스트 node id 실행\n"
	@printf "  make test-k K=...              -k 필터 실행\n"
	@printf "  make typecheck                 mypy 실행\n"
	@printf "  make lint                      ruff 실행\n"
	@printf "  make test-cov                  커버리지 포함 전체 테스트\n"
	@printf "  make check                     lint + typecheck + test\n"
	@printf "  make help-tests                기능 타깃별 포함 테스트 파일 목록\n"

help-tests:
	@printf "기능별 모듈 타깃:\n"
	@printf "  test-module_config       -> $(MODULE_config)\n"
	@printf "  test-module_adapters     -> $(MODULE_adapters)\n"
	@printf "  test-module_app_bridge   -> $(MODULE_app_bridge)\n"
	@printf "  test-module_pipeline     -> $(MODULE_pipeline)\n"
	@printf "  test-module_diffusion    -> $(MODULE_diffusion)\n"
	@printf "  test-module_oxidation    -> $(MODULE_oxidation)\n"
	@printf "  test-module_metrics      -> $(MODULE_metrics)\n"
	@printf "  test-module_export       -> $(MODULE_export)\n"
	@printf "  test-module_quality      -> $(MODULE_quality)\n"
	@printf "\n"
	@printf "통합 타깃:\n"
	@printf "  test-integration_all     -> marker=integration\n"
	@printf "  test-integration_pipeline-> $(INTEGRATION_pipeline)\n"
	@printf "  test-integration_oxidation -> $(INTEGRATION_oxidation)\n"
	@printf "  test-integration_gui     -> $(INTEGRATION_gui)\n"
	@printf "  test-integration_history -> $(INTEGRATION_history)\n"

test:
	$(PYTEST) $(PYTEST_PRETTY_OPTS) $(PYTEST_EXTRA_OPTS) tests

test-all: test

test-fast:
	$(PYTEST) $(PYTEST_PRETTY_OPTS) $(PYTEST_EXTRA_OPTS) tests -x --maxfail=1

test-module_config:
	$(PYTEST) $(PYTEST_PRETTY_OPTS) $(PYTEST_EXTRA_OPTS) $(MODULE_config)

test-module_adapters:
	$(PYTEST) $(PYTEST_PRETTY_OPTS) $(PYTEST_EXTRA_OPTS) $(MODULE_adapters)

test-module_app_bridge:
	$(PYTEST) $(PYTEST_PRETTY_OPTS) $(PYTEST_EXTRA_OPTS) $(MODULE_app_bridge)

test-module_pipeline:
	$(PYTEST) $(PYTEST_PRETTY_OPTS) $(PYTEST_EXTRA_OPTS) $(MODULE_pipeline)

test-module_diffusion:
	$(PYTEST) $(PYTEST_PRETTY_OPTS) $(PYTEST_EXTRA_OPTS) $(MODULE_diffusion)

test-module_oxidation:
	$(PYTEST) $(PYTEST_PRETTY_OPTS) $(PYTEST_EXTRA_OPTS) $(MODULE_oxidation)

test-module_metrics:
	$(PYTEST) $(PYTEST_PRETTY_OPTS) $(PYTEST_EXTRA_OPTS) $(MODULE_metrics)

test-module_export:
	$(PYTEST) $(PYTEST_PRETTY_OPTS) $(PYTEST_EXTRA_OPTS) $(MODULE_export)

test-module_quality:
	$(PYTEST) $(PYTEST_PRETTY_OPTS) $(PYTEST_EXTRA_OPTS) $(MODULE_quality)

test-integration_all:
	$(PYTEST) $(PYTEST_PRETTY_OPTS) $(PYTEST_EXTRA_OPTS) -m "integration"

test-integration_pipeline:
	$(PYTEST) $(PYTEST_PRETTY_OPTS) $(PYTEST_EXTRA_OPTS) -m "integration" $(INTEGRATION_pipeline)

test-integration_oxidation:
	$(PYTEST) $(PYTEST_PRETTY_OPTS) $(PYTEST_EXTRA_OPTS) -m "integration" $(INTEGRATION_oxidation)

test-integration_gui:
	$(PYTEST) $(PYTEST_PRETTY_OPTS) $(PYTEST_EXTRA_OPTS) -m "integration" $(INTEGRATION_gui)

test-integration_history:
	$(PYTEST) $(PYTEST_PRETTY_OPTS) $(PYTEST_EXTRA_OPTS) -m "integration" $(INTEGRATION_history)

test-one:
	@if [ -z "$(TEST)" ]; then printf "Usage: make test-one TEST=tests/...::test_name\n"; exit 1; fi
	$(PYTEST) $(PYTEST_PRETTY_OPTS) $(PYTEST_EXTRA_OPTS) "$(TEST)"

test-k:
	@if [ -z "$(K)" ]; then printf "Usage: make test-k K=<expr>\n"; exit 1; fi
	$(PYTEST) $(PYTEST_PRETTY_OPTS) $(PYTEST_EXTRA_OPTS) -k "$(K)"

typecheck:
	$(MYPY)

lint:
	$(RUFF) check proc2d tests

test-cov:
	$(PYTEST) $(PYTEST_EXTRA_OPTS) tests --cov=proc2d --cov-report=term-missing --cov-report=xml

check: typecheck lint test

refactor-check: check
