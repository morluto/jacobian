.DEFAULT_GOAL := help

UV_RUN := uv run --locked
PYTEST_ARGS ?=
SCALE_WORKERS ?= 2
TESTS ?=
PATHS ?=
AFFECTED_BASE ?= origin/main
JUNIT ?= pytest.xml
TIMING ?=
TIMING_LIMIT ?= 20
ALLOW_PARALLEL_VALIDATION ?= 0
EVAL_ARGS ?=
STRESS_COUNT ?= 3
ORDERING_DEFAULT_SEED := --randomly-seed=17
PYTEST_DIAGNOSTIC_ARGS ?= --durations=10
ORDINARY_MARKER_EXPRESSION := not property and not exhaustive and not scale
RUFF_PATHS := src tests benchmarks typings
PYTEST_RUNNER := $(UV_RUN) python tools/pytest_lifecycle.py
VALIDATION_LOCK := $(UV_RUN) python tools/with_validation_lock.py
# Owner lanes cover every ordinary test root exactly once. CI runs
# them independently; `make check-all` reproduces them locally in this order.
ORDINARY_TEST_LANES := math catalog dispatch cli tooling integration
FOCUSED_TEST_LANES := $(ORDINARY_TEST_LANES) process mcp
PUBLIC_COMMANDS := setup affected handoff-scoped test-focused quick-scoped affected-plan test-timings check check-all fix

include make/development.mk
include make/harbor.mk
include make/evaluations.mk

# Timeouts are per-command, not pyproject addopts: process lanes isolate
# killable work, and a global signal deadline would hit native solvers.
.PHONY: help help-all

help: ## Show the primary developer workflow.
	@awk -v public="$(PUBLIC_COMMANDS)" 'BEGIN {FS = ":.*## "; n = split(public, names, " "); for (i = 1; i <= n; i++) wanted[names[i]] = 1; printf "Jacobian primary developer commands:\n\n"} /^[a-zA-Z0-9_-]+:.*## / && ($$1 in wanted) {description[$$1] = $$2} END {for (i = 1; i <= n; i++) printf "  %-18s %s\n", names[i], description[names[i]]}' $(MAKEFILE_LIST)
	@printf '\nAdvanced lifecycle and diagnostic commands are hidden from the daily index. Use `make help-all` to list them.\n'

help-all: ## Show every low-level and lifecycle developer command.
	@awk 'BEGIN {FS = ":.*## "; printf "All Jacobian developer commands:\n\n"} /^[a-zA-Z0-9_-]+:.*## / {printf "  %-26s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

test-math: ## Ordinary domain-owned mathematical behavior (1 worker, 120s).
	$(UV_RUN) pytest -n 1 --dist worksteal --timeout=120 \
		-m "$(ORDINARY_MARKER_EXPRESSION)" $(if $(TESTS),$(TESTS),tests/math) \
		$(PYTEST_DIAGNOSTIC_ARGS) $(PYTEST_ARGS)

test-catalog: ## Immutable catalog and discovery behavior (2 workers, 30s).
	$(UV_RUN) pytest -n 2 --dist worksteal --timeout=30 \
		$(if $(TESTS),$(TESTS),tests/catalog) \
		$(PYTEST_DIAGNOSTIC_ARGS) $(PYTEST_ARGS)

test-dispatch: ## Strict parsing and direct dispatch behavior (2 workers, 120s).
	$(UV_RUN) pytest -n 2 --dist worksteal --timeout=120 \
		$(if $(TESTS),$(TESTS),tests/dispatch) \
		$(PYTEST_DIAGNOSTIC_ARGS) $(PYTEST_ARGS)

test-cli: ## Command-line boundary behavior (2 workers, 30s).
	$(UV_RUN) pytest -n 2 --dist worksteal --timeout=30 \
		$(if $(TESTS),$(TESTS),tests/cli) \
		$(PYTEST_DIAGNOSTIC_ARGS) $(PYTEST_ARGS)

test-tooling: ## Repository tooling and static contracts (2 workers, 30s).
	$(UV_RUN) pytest -n 2 --dist worksteal --timeout=30 \
		$(if $(TESTS),$(TESTS),tests/tooling) \
		$(PYTEST_DIAGNOSTIC_ARGS) $(PYTEST_ARGS)

test-integration: ## Ordinary cross-owner mathematical seams (2 workers, 120s).
	$(UV_RUN) pytest -n 1 --dist worksteal --timeout=120 -m "$(ORDINARY_MARKER_EXPRESSION)" \
		$(if $(TESTS),$(TESTS),tests/integration --ignore=tests/integration/catalog) \
		$(PYTEST_DIAGNOSTIC_ARGS) $(PYTEST_ARGS)

test-focused: ## Run TESTS through its explicit semantic LANE (for example, LANE=math).
	@test -n "$(LANE)" || { echo "LANE is required, e.g. LANE=math" >&2; exit 2; }
	@test -n "$(TESTS)" || { echo "TESTS is required, e.g. TESTS=tests/math/..." >&2; exit 2; }
	@case " $(FOCUSED_TEST_LANES) " in *" $(LANE) "*) ;; *) \
		echo "LANE must be one of: $(FOCUSED_TEST_LANES)" >&2; exit 2;; esac
	$(MAKE) test-$(LANE) TESTS="$(TESTS)" PYTEST_ARGS="$(PYTEST_ARGS)"

affected: ## Default local validation: CI-planned affected owners and scoped static checks.
	$(UV_RUN) python tools/affected_validation.py --base "$(AFFECTED_BASE)"

affected-plan: ## Show the CI-planned local validation selected from AFFECTED_BASE...HEAD.
	$(UV_RUN) python tools/affected_validation.py --base "$(AFFECTED_BASE)" --dry-run

test-timings: ## Summarize pytest JUnit timing evidence; set TIMING for worker skew.
	$(UV_RUN) python tools/test_timing_report.py --junit "$(JUNIT)" \
		$(if $(TIMING),--timing "$(TIMING)") --limit "$(TIMING_LIMIT)"

lint-scoped: ## Check explicit Python PATHS with Ruff without touching unrelated files.
	@test -n "$(PATHS)" || { echo "PATHS is required, e.g. PATHS='src/jacobian/... tests/math/...'" >&2; exit 2; }
	$(UV_RUN) ruff check $(PATHS)
	$(UV_RUN) ruff format --check $(PATHS)

typecheck-scoped: ## Type-check explicit Python PATHS and their imported dependencies.
	@test -n "$(PATHS)" || { echo "PATHS is required, e.g. PATHS='src/jacobian/... tests/math/...'" >&2; exit 2; }
	$(UV_RUN) mypy $(PATHS)

test-fast: ## Broad ordinary owner tests except cross-owner integration.
	# Full catalog construction imports every maintained math backend; keep
	# the fast lane within hosted-runner memory instead of crashing workers.
	$(UV_RUN) pytest -n 2 --dist worksteal --timeout=120 -m "$(ORDINARY_MARKER_EXPRESSION)" \
		$(if $(TESTS),$(TESTS),tests/math tests/catalog tests/dispatch tests/cli tests/tooling) \
		$(PYTEST_DIAGNOSTIC_ARGS) $(PYTEST_ARGS)

test-process: ## Killable child-process boundaries (2 workers, 120s).
	$(PYTEST_RUNNER) --name process --timeout-seconds 4800 -- \
		-n 2 --dist worksteal --timeout=120 --timeout-method=signal \
		$(if $(TESTS),$(TESTS),tests/process) \
		$(PYTEST_DIAGNOSTIC_ARGS) $(PYTEST_ARGS)

test-mcp: ## MCP transport boundaries (2 workers, 120s).
	$(PYTEST_RUNNER) --name mcp --timeout-seconds 4800 -- \
		-n 2 --dist worksteal --timeout=120 --timeout-method=signal \
		$(if $(TESTS),$(TESTS),tests/mcp) \
		$(PYTEST_DIAGNOSTIC_ARGS) $(PYTEST_ARGS)

test-singular: ## Pinned Singular exact-algebra backend (serial, 120s, kill-safe).
	@command -v Singular >/dev/null || { echo "Singular 4.4 is required" >&2; exit 1; }
	$(PYTEST_RUNNER) --name singular --timeout-seconds 1200 -- \
		-n 0 --timeout=120 --timeout-method=signal \
		tests/math/polynomials/ideals \
	tests/math/polynomials/test_polynomial_map_generic_degree.py \
	tests/process/polynomials/ideals \
	tests/process/polynomial_maps \
	tests/integration/catalog/test_builtin_examples.py \
	tests/integration/catalog/test_mcp_builtin_examples.py \
		$(PYTEST_DIAGNOSTIC_ARGS) $(PYTEST_ARGS)

test-qepcad: ## Pinned QEPCAD plane-topology backend (serial, kill-safe).
	@command -v qepcad >/dev/null || { echo "QEPCAD 1.74 is required" >&2; exit 1; }
	@qepcad -v | grep -F "Version B 1.74," >/dev/null || { echo "QEPCAD 1.74 is required" >&2; exit 1; }
	$(PYTEST_RUNNER) --name qepcad --timeout-seconds 1800 -- \
		-n 0 --timeout=600 --timeout-method=signal \
		tests/math/polynomials/test_plane_component_profile.py \
		tests/process/polynomials/test_qepcad_plane_components.py \
		$(PYTEST_DIAGNOSTIC_ARGS) $(PYTEST_ARGS)

test: test-ordinary ## All ordinary Python tests.

test-ordinary: ## Ordinary suite in the fixed CI group order.
	@for lane in $(ORDINARY_TEST_LANES); do \
		$(MAKE) test-$$lane || exit $$?; \
	done

test-compatibility: ## Supported-version import/API compatibility smoke.
	$(UV_RUN) pytest -n 0 --timeout=30 --timeout-method=thread \
		tests/tooling/test_ci_compatibility.py \
		$(PYTEST_DIAGNOSTIC_ARGS) $(PYTEST_ARGS)

test-full: ## Every local semantic pytest lane; not hosted CI, coverage, or docs.
	$(VALIDATION_LOCK) run --target test-full -- $(MAKE) _test-full

_test-full:
	$(MAKE) test-math
	$(MAKE) test-property
	$(MAKE) _test-exhaustive
	$(MAKE) _test-scale
	$(MAKE) test-catalog
	$(MAKE) test-dispatch
	$(MAKE) test-cli
	$(MAKE) test-tooling
	$(MAKE) test-integration
	$(MAKE) test-process
	$(MAKE) test-mcp
	$(MAKE) test-singular
	$(MAKE) test-qepcad

test-property: ## Run explicitly marked invariant checks once.
	$(UV_RUN) pytest -n 0 --timeout=120 --timeout-method=thread -m property \
		$(if $(TESTS),$(TESTS),tests) \
		$(PYTEST_DIAGNOSTIC_ARGS) $(PYTEST_ARGS)

test-stress: ## Repeat explicitly marked property tests on the scheduled lane.
	$(MAKE) test-property PYTEST_ARGS="--count=$(STRESS_COUNT) $(PYTEST_ARGS)"

test-scale: ## Run optional near-envelope mathematical execution evidence ($(SCALE_WORKERS) workers).
	$(VALIDATION_LOCK) run --target test-scale -- $(MAKE) _test-scale

_test-scale:
	$(UV_RUN) pytest -n $(SCALE_WORKERS) --dist worksteal --timeout=180 --timeout-method=thread -m scale \
		$(if $(TESTS),$(TESTS),tests) \
		$(PYTEST_DIAGNOSTIC_ARGS) $(PYTEST_ARGS)

test-exhaustive: ## Broad finite reference sweeps reserved for scheduled validation (2 workers).
	$(VALIDATION_LOCK) run --target test-exhaustive -- $(MAKE) _test-exhaustive

_test-exhaustive:
	$(UV_RUN) pytest -n 2 --dist worksteal --timeout=180 --timeout-method=thread -m exhaustive \
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
	@violations="$$(grep -R -n --include='*.py' 'TODO' src/ tests/ | grep -E -v 'TODO\(#[0-9]+\)' || true)"; \
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
	$(UV_RUN) python tools/check_wheel_contents.py

handoff: lint typecheck test-focused ## Focused contributor handoff: lint, types, and one declared owner path.

handoff-scoped: lint-scoped typecheck-scoped test-focused ## Focused handoff scoped to declared static PATHS and one owner test path.

quick: lint test-focused ## Broad Ruff checks plus one declared owner test path.

quick-scoped: lint-scoped test-focused ## Focused edit loop scoped to declared static PATHS and one owner test path.

check: ## Final broad gate: lint, types, and all non-integration owner tests.
ifeq ($(ALLOW_PARALLEL_VALIDATION),1)
	$(MAKE) _check
else
	$(VALIDATION_LOCK) run --target check -- $(MAKE) _check
endif

_check: lint typecheck test-fast

check-all: ## Escalation: reproduce all ordinary Python CI lanes locally.
ifeq ($(ALLOW_PARALLEL_VALIDATION),1)
	$(MAKE) _check-all
else
	$(VALIDATION_LOCK) run --target check-all -- $(MAKE) _check-all
endif

_check-all: lint typecheck test-ordinary

precommit: ## Apply safe fixes, then run the broad ordinary gate (mutates the tree).
	$(MAKE) fix
	$(MAKE) check

validation-status: ## Show whether this worktree holds a broad-validation lock.
	$(VALIDATION_LOCK) status

check-static: lint-full typecheck import-contracts architecture todo-check build ## CI-owned static checks plus a local package build.

clean: ## Remove local caches, build outputs, and coverage artifacts.
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build htmlcov
	rm -f .coverage .coverage.*
	find src tests benchmarks typings -type d -name '__pycache__' -prune -exec rm -rf {} +
	find . -maxdepth 2 -type d -name '*.egg-info' -prune -exec rm -rf {} +
