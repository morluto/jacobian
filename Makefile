.DEFAULT_GOAL := help

UV_RUN := uv run --locked
PYTEST_ARGS ?=
TESTS ?=
EVAL_ARGS ?=
STRESS_COUNT ?= 3
ORDERING_DEFAULT_SEED := --randomly-seed=17
PYTEST_DIAGNOSTIC_ARGS ?= --durations=10
RUFF_PATHS := src tests benchmarks
TOPOLOGY_RUNNER := $(UV_RUN) python tools/test_topology.py
PYTEST_RUNNER := $(UV_RUN) python tools/pytest_lifecycle.py
PUBLIC_COMMANDS := help setup check check-changed ci-plan test-plan test-changed test-unit test-component test-domain test-composition test-storage test-process test-mcp test-provider test-lean test-e2e docs-command-check docs-linkcheck harbor-plan harbor-prepare-task harbor-validate-task harbor-execution-check harbor-check-task harbor-oracle-task npm-test test-all-ci check-static deploy-check

ifneq ($(strip $(PATHS)),)
PATHS_FILE := $(shell mktemp)
$(file >$(PATHS_FILE),$(PATHS))
endif

include make/development.mk
include make/harbor.mk
include make/evaluations.mk

# A timeout is a lane-level containment policy.  It intentionally does not live
# in pyproject.toml: direct pytest invocations must not silently inherit a
# signal-based deadline that cannot interrupt a native solver.  Process and
# provider lanes run risky work in killable children and set their own deadline.
.PHONY: help help-all

help: ## Show available developer commands.
	@awk -v public="$(PUBLIC_COMMANDS)" 'BEGIN {FS = ":.*## "; n = split(public, names, " "); for (i = 1; i <= n; i++) wanted[names[i]] = 1; printf "Jacobian common developer commands:\n\n"} /^[a-zA-Z_-]+:.*## / && ($$1 in wanted) {printf "  %-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@printf '\nAdvanced lifecycle and diagnostic commands are hidden from the daily index. Use `make help-all` to list them.\n'

help-all: ## Show every low-level and lifecycle developer command.
	@awk 'BEGIN {FS = ":.*## "; printf "All Jacobian developer commands:\n\n"} /^[a-zA-Z_-]+:.*## / {printf "  %-26s %s\n", $$1, $$2}' $(MAKEFILE_LIST)


test-plan: ## Print local validation selected for BASE..HEAD or explicit PATHS.
	@if [ -n "$(PATHS)" ]; then \
		trap 'rm -f "$(PATHS_FILE)"' EXIT; \
		$(UV_RUN) python .github/scripts/plan-local-tests --paths-file "$(PATHS_FILE)"; \
	else \
		test -n "$(BASE)" || { echo "BASE is required unless PATHS is set (for example: make test-plan BASE=origin/main)" >&2; exit 2; }; \
		$(UV_RUN) python .github/scripts/plan-local-tests --base "$(BASE)"; \
	fi

ci-plan: ## Print the hosted CI lane plan for BASE..HEAD and working changes.
	@set -eu; \
	tmp_dir=$$(mktemp -d); \
	trap 'rm -rf "$$tmp_dir"; if [ -n "$(PATHS_FILE)" ]; then rm -f "$(PATHS_FILE)"; fi' EXIT; \
	base_sha=$$(git rev-parse "$(or $(BASE),origin/main)"); \
	head_sha=$$(git rev-parse HEAD); \
	if [ -n "$(PATHS)" ]; then \
		$(UV_RUN) python .github/scripts/normalize-ci-paths --file "$(PATHS_FILE)" > "$$tmp_dir/changed-paths.txt"; \
	else \
		{ \
			git diff --name-only "$(or $(BASE),origin/main)" HEAD; \
			git diff --name-only HEAD; \
			git diff --cached --name-only; \
			git ls-files --others --exclude-standard; \
		} | sort -u > "$$tmp_dir/changed-paths.txt"; \
	fi; \
	if [ -n "$(PATHS)" ]; then \
		$(UV_RUN) python .github/scripts/classify-ci-paths --paths-file "$(PATHS_FILE)" > "$$tmp_dir/plan.txt"; \
	else \
		changed_paths=$$(tr '\n' ' ' < "$$tmp_dir/changed-paths.txt"); \
		$(UV_RUN) python .github/scripts/classify-ci-paths -- $$changed_paths > "$$tmp_dir/plan.txt"; \
	fi; \
	$(UV_RUN) python .github/scripts/validate-ci-plan < "$$tmp_dir/plan.txt"; \
	$(UV_RUN) python .github/scripts/emit-plan-receipt \
		--kind product-ci --event pull_request \
		--base "$$base_sha" --head "$$head_sha" \
		--planner .github/scripts/classify-ci-paths \
		--config .github/ci-impact.json --config tests/topology.toml \
		--config .github/scripts/_ci_paths.py \
		--config .github/scripts/validate-ci-plan \
		--config .github/workflows/ci.yml --config Makefile \
		--config make/development.mk --config make/harbor.mk \
		--config make/evaluations.mk \
		--plan-file "$$tmp_dir/plan.txt" \
		--paths-file "$$tmp_dir/changed-paths.txt" \
		--output "$$tmp_dir/receipt.json" >/dev/null; \
	echo "Hosted CI lanes:"; \
	cat "$$tmp_dir/plan.txt"; \
	echo "Plan receipt:"; \
	cat "$$tmp_dir/receipt.json"

test-changed: ## Run changed-path tests, defaulting BASE to origin/main.
	@if [ -n "$(PATHS)" ]; then \
		trap 'rm -f "$(PATHS_FILE)"' EXIT; \
		$(UV_RUN) python .github/scripts/plan-local-tests --paths-file "$(PATHS_FILE)" --execute; \
	else \
		$(UV_RUN) python .github/scripts/plan-local-tests --base "$(or $(BASE),origin/main)" --execute; \
	fi

check-changed: ## Run format, types, and exact changed-path tests.
	$(MAKE) lint typecheck
	$(MAKE) test-changed BASE="$(or $(BASE),origin/main)"

define run_topology_lane
	$(TOPOLOGY_RUNNER) $(1) \
		--pytest-args "$(PYTEST_DIAGNOSTIC_ARGS) $(PYTEST_ARGS)" \
		$(if $(TESTS),$(TESTS))
endef

test-unit: ## Run pure contracts and models (10s lane, sequential).
	$(call run_topology_lane,unit)

test-component: ## Run one-service component tests (30s lane, four workers).
	$(call run_topology_lane,component)

test-domain: ## Run explicitly bundled mathematical domains (120s lane).
	$(call run_topology_lane,domain)

test-composition: ## Run complete-runtime composition tests (120s, two workers).
	$(call run_topology_lane,composition)

test-storage: ## Run SQLite durability and recovery boundaries (serial).
	$(call run_topology_lane,storage)

test-process: ## Run killable child-process boundaries (two workers).
	$(call run_topology_lane,process)

test-mcp: ## Run MCP transport boundaries (two workers).
	$(call run_topology_lane,mcp)

test-provider: ## Run prepared optional-provider boundaries (one worker).
	$(call run_topology_lane,provider)

test-lean: ## Run the pinned Lean/Mathlib boundary serially.
	$(call run_topology_lane,lean)

test-e2e: ## Run complete user-visible CLI/workflow scenarios serially.
	$(call run_topology_lane,e2e)

test-compatibility: ## Run the small supported-version import/API compatibility smoke suite.
	$(PYTEST_RUNNER) --name compatibility -- -n 0 --timeout=30 --timeout-method=thread tests/unit/tooling/test_ci_compatibility.py $(PYTEST_DIAGNOSTIC_ARGS) $(PYTEST_ARGS)

test-affected: test-changed ## Compatibility alias for the changed-path planner.

test-all-ci: ## Explicitly run every semantic lane locally (exceptional).
	$(MAKE) test-unit
	$(MAKE) test-component
	$(MAKE) test-domain
	$(MAKE) test-composition
	$(MAKE) test-storage
	$(MAKE) test-process
	$(MAKE) test-mcp
	$(MAKE) test-provider
	$(MAKE) test-lean
	$(MAKE) test-e2e

test-stress: ## Repeat explicitly marked property tests on the scheduled lane.
	$(PYTEST_RUNNER) --name stress -- -n 0 --timeout=120 --timeout-method=thread -m property --count=$(STRESS_COUNT) \
		$(if $(TESTS),$(TESTS),tests) $(PYTEST_DIAGNOSTIC_ARGS) $(PYTEST_ARGS)

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

check: lint typecheck test-unit ## Run the fast routine local handoff checks.

precommit: ## Fix and run every routine local handoff check.
	$(MAKE) fix
	$(MAKE) check

check-static: lint-full typecheck test-architecture architecture todo-check build ## Run CI-owned static checks plus a local package build.

clean: ## Remove local caches, build outputs, and coverage artifacts.
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build htmlcov
	rm -f .coverage .coverage.*
	find src tests benchmarks -type d -name '__pycache__' -prune -exec rm -rf {} +
	find . -maxdepth 2 -type d -name '*.egg-info' -prune -exec rm -rf {} +
