.DEFAULT_GOAL := help

UV_RUN := uv run --locked
PYTEST_ARGS ?=
TESTS ?=
EVAL_ARGS ?=
STRESS_COUNT ?= 3
ORDERING_DEFAULT_SEED := --randomly-seed=17
PYTEST_DIAGNOSTIC_ARGS ?= --durations=10
RUFF_PATHS := src tests benchmarks
PYTEST_RUNNER := $(UV_RUN) python tools/pytest_lifecycle.py
# Fixed semantic lanes covering the Lean-free ordinary testpaths. CI runs these
# independently; `make check` runs the same lanes locally in this order.
ORDINARY_TEST_LANES := unit component domain composition e2e provider
PUBLIC_COMMANDS := setup quick check check-external fix

include make/development.mk
include make/harbor.mk
include make/evaluations.mk

# Timeouts are per-command, not pyproject addopts: process/Lean isolate
# killable work, and a global signal deadline would hit native solvers.
.PHONY: help help-all

help: ## Show the primary developer workflow.
	@awk -v public="$(PUBLIC_COMMANDS)" 'BEGIN {FS = ":.*## "; n = split(public, names, " "); for (i = 1; i <= n; i++) wanted[names[i]] = 1; printf "Jacobian primary developer commands:\n\n"} /^[a-zA-Z0-9_-]+:.*## / && ($$1 in wanted) {description[$$1] = $$2} END {for (i = 1; i <= n; i++) printf "  %-18s %s\n", names[i], description[names[i]]}' $(MAKEFILE_LIST)
	@printf '\nAdvanced lifecycle and diagnostic commands are hidden from the daily index. Use `make help-all` to list them.\n'

help-all: ## Show every low-level and lifecycle developer command.
	@awk 'BEGIN {FS = ":.*## "; printf "All Jacobian developer commands:\n\n"} /^[a-zA-Z0-9_-]+:.*## / {printf "  %-26s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

test-unit: ## Pure contracts and models (sequential, 10s).
	$(UV_RUN) pytest --timeout=10 \
		$(if $(TESTS),$(TESTS),tests/unit) \
		$(PYTEST_DIAGNOSTIC_ARGS) $(PYTEST_ARGS)

test-component: ## One-service component tests (4 workers, 30s).
	$(UV_RUN) pytest -n 4 --dist worksteal --timeout=30 -m "not exhaustive" \
		$(if $(TESTS),$(TESTS),tests/component) \
		$(PYTEST_DIAGNOSTIC_ARGS) $(PYTEST_ARGS)

test-domain: ## Explicit mathematical domains (4 workers, 120s).
	$(UV_RUN) pytest -n 4 --dist worksteal --timeout=120 \
		$(if $(TESTS),$(TESTS),tests/domain) \
		$(PYTEST_DIAGNOSTIC_ARGS) $(PYTEST_ARGS)

test-composition: ## Cross-domain composition (2 workers, 120s).
	$(UV_RUN) pytest -n 2 --dist worksteal --timeout=120 \
		$(if $(TESTS),$(TESTS),tests/composition) \
		$(PYTEST_DIAGNOSTIC_ARGS) $(PYTEST_ARGS)

test-storage: ## SQLite durability and recovery (serial, 120s).
	$(UV_RUN) pytest --timeout=120 \
		$(if $(TESTS),$(TESTS),tests/boundary/storage) \
		$(PYTEST_DIAGNOSTIC_ARGS) $(PYTEST_ARGS)

test-process: ## Killable child-process boundaries (2 workers, 120s).
	$(PYTEST_RUNNER) --name process --timeout-seconds 4800 -- \
		-n 2 --dist worksteal --timeout=120 --timeout-method=signal \
		$(if $(TESTS),$(TESTS),tests/boundary/process) \
		$(PYTEST_DIAGNOSTIC_ARGS) $(PYTEST_ARGS)

test-mcp: ## MCP transport boundaries (2 workers, 120s).
	$(PYTEST_RUNNER) --name mcp --timeout-seconds 4800 -- \
		-n 2 --dist worksteal --timeout=120 --timeout-method=signal \
		$(if $(TESTS),$(TESTS),tests/boundary/mcp) \
		$(PYTEST_DIAGNOSTIC_ARGS) $(PYTEST_ARGS)

test-provider: ## Maintained Python provider boundaries (1 worker, 180s).
	$(UV_RUN) pytest -n 1 --dist load --timeout=180 \
		$(if $(TESTS),$(TESTS),tests/boundary/providers/cvc5 tests/boundary/providers/external_sat tests/boundary/providers/flint) \
		$(PYTEST_DIAGNOSTIC_ARGS) $(PYTEST_ARGS)

test-lean: ## Pinned Lean/Mathlib boundary (serial, 300s, kill-safe).
	$(PYTEST_RUNNER) --name lean --timeout-seconds 12000 -- \
		--timeout=300 --timeout-method=signal \
		$(if $(TESTS),$(TESTS),tests/boundary/providers/lean) \
		$(PYTEST_DIAGNOSTIC_ARGS) $(PYTEST_ARGS)

test-e2e: ## Complete caller-visible journeys (serial, 180s).
	$(UV_RUN) pytest --timeout=180 \
		$(if $(TESTS),$(TESTS),tests/e2e) \
		$(PYTEST_DIAGNOSTIC_ARGS) $(PYTEST_ARGS)

test-ordinary: ## Lean-free ordinary suite in the fixed CI group order.
	@for lane in $(ORDINARY_TEST_LANES); do \
		$(MAKE) test-$$lane || exit $$?; \
	done

test-compatibility: ## Supported-version import/API compatibility smoke.
	$(UV_RUN) pytest -n 0 --timeout=30 --timeout-method=thread \
		tests/unit/tooling/test_ci_compatibility.py \
		$(PYTEST_DIAGNOSTIC_ARGS) $(PYTEST_ARGS)

test-all-ci: ## Explicitly run every semantic lane locally (exceptional).
	$(MAKE) test-unit
	$(MAKE) test-component
	$(MAKE) test-exhaustive
	$(MAKE) test-domain
	$(MAKE) test-composition
	$(MAKE) test-storage
	$(MAKE) test-process
	$(MAKE) test-mcp
	$(MAKE) test-provider
	$(MAKE) test-lean
	$(MAKE) test-e2e

test-stress: ## Repeat explicitly marked property tests on the scheduled lane.
	$(UV_RUN) pytest -n 0 --timeout=120 --timeout-method=thread -m property \
		--count=$(STRESS_COUNT) $(if $(TESTS),$(TESTS),tests) \
		$(PYTEST_DIAGNOSTIC_ARGS) $(PYTEST_ARGS)

test-exhaustive: ## Broad finite reference sweeps reserved for scheduled validation.
	$(UV_RUN) pytest -n 0 --timeout=180 --timeout-method=thread -m exhaustive \
		$(if $(TESTS),$(TESTS),tests) \
		$(PYTEST_DIAGNOSTIC_ARGS) $(PYTEST_ARGS)

test-ordering: ## Reproduce scheduled ordering (default seed 17; override with PYTEST_ARGS).
	@test -n "$(ORDERING_LANE)" || { echo "ORDERING_LANE is required" >&2; exit 2; }
	$(MAKE) test-$(ORDERING_LANE) \
		PYTEST_ARGS="$(if $(findstring --randomly-seed,$(PYTEST_ARGS)),,$(ORDERING_DEFAULT_SEED)) $(PYTEST_ARGS)"

duplicate-code: ## Run the CI duplicate-code detector locally.
	npx --yes jscpd@5.0.12 --config .jscpd.json .

npm-test: ## Run the npm package tests and dry-run pack.
	npm test --prefix npm
	npm pack --dry-run ./npm

todo-check: ## Fail on TODO comments that do not reference an issue.
	@violations="$$(rg -n 'TODO' --type py src/ tests/ | rg -v 'TODO\(#\d+\)' || true)"; \
	if [ -n "$$violations" ]; then \
	  printf '%s\n' "$$violations"; \
	  echo "TODO comments must reference an issue, e.g. TODO(#123)." >&2; \
	  exit 1; \
	fi

coverage: ## Combine coverage data files and enforce the repository threshold.
	$(UV_RUN) coverage combine
	$(UV_RUN) coverage report --fail-under=50
	$(UV_RUN) coverage xml

build: ## Build Python source and wheel distributions.
	uv build

quick: lint typecheck test-unit ## Cheap iteration: lint, types, unit tests.

check: lint typecheck test-ordinary ## PR-equivalent ordinary Python validation.

check-external: test-lean test-provider ## Lean and maintained-provider isolation.

precommit: ## Fix and run every routine local handoff check.
	$(MAKE) fix
	$(MAKE) quick

check-static: lint-full typecheck import-contracts architecture todo-check build ## CI-owned static checks plus a local package build.

clean: ## Remove local caches, build outputs, and coverage artifacts.
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build htmlcov
	rm -f .coverage .coverage.*
	find src tests benchmarks -type d -name '__pycache__' -prune -exec rm -rf {} +
	find . -maxdepth 2 -type d -name '*.egg-info' -prune -exec rm -rf {} +
