.DEFAULT_GOAL := help

UV_RUN := uv run --locked
HARBOR_VERSION ?= 0.20.0
HARBOR_RUNNER ?= uvx --from harbor==$(HARBOR_VERSION) harbor
HARBOR_PYTHON ?= uvx --from harbor==$(HARBOR_VERSION) python
PYTEST_ARGS ?=
TESTS ?=
EVAL_ARGS ?=
STRESS_COUNT ?= 3
ORDERING_DEFAULT_SEED := --randomly-seed=17
PYTEST_DIAGNOSTIC_ARGS ?= --durations=10
RUFF_PATHS := src tests benchmarks
TOPOLOGY_RUNNER := $(UV_RUN) python tools/test_topology.py

# A timeout is a lane-level containment policy.  It intentionally does not live
# in pyproject.toml: direct pytest invocations must not silently inherit a
# signal-based deadline that cannot interrupt a native solver.  Process and
# provider lanes run risky work in killable children and set their own deadline.
.PHONY: help setup hooks fix lint complexity-check lint-full security-audit typecheck test-architecture test-plan test-changed test-unit test-component test-domain test-composition test-storage test-process test-mcp test-provider test-lean test-e2e test-affected test-all-ci test-compatibility test-stress test-ordering duplicate-code npm-test todo-check coverage build check precommit check-static harbor-check harbor-oracle agent-eval bench-core clean docs-linkcheck deploy-check

help: ## Show available developer commands.
	@awk 'BEGIN {FS = ":.*## "; printf "Jacobian developer commands:\n\n"} /^[a-zA-Z_-]+:.*## / {printf "  %-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Install the locked development environment.
	uv sync --locked --dev

deploy-check: ## Validate the clone-to-systemd deployment entrypoint.
	bash -n deploy/install.sh
	$(UV_RUN) pytest -n 0 tests/boundary/process/tooling/test_deploy_installer.py

hooks: setup ## Install pre-commit hooks.
	$(UV_RUN) pre-commit install --install-hooks
	$(UV_RUN) pre-commit install --hook-type pre-push

fix: ## Apply Ruff fixes and formatting.
	$(UV_RUN) ruff check --fix $(RUFF_PATHS)
	$(UV_RUN) ruff format $(RUFF_PATHS)

lint: ## Run the fast Ruff lint and format checks.
	$(UV_RUN) ruff check $(RUFF_PATHS)
	$(UV_RUN) ruff format --check $(RUFF_PATHS)
	$(MAKE) complexity-check

complexity-check: ## Reject new, increased, or stale C901 baseline entries.
	$(UV_RUN) python tools/check_complexity.py

lint-full: lint ## Add dependency and dead-code checks.
	$(UV_RUN) deptry .
	$(UV_RUN) vulture src tests --min-confidence=80

security-audit: ## Audit dependencies for known vulnerabilities.
	$(UV_RUN) pip-audit

typecheck: ## Run strict static type checking.
	$(UV_RUN) mypy

test-architecture: ## Enforce semantic test-layer and provider-import boundaries.
	$(UV_RUN) python tools/check_test_architecture.py .

test-plan: ## Print local validation selected for BASE..HEAD and working changes.
	@test -n "$(BASE)" || { echo "BASE is required (for example: make test-plan BASE=origin/main)" >&2; exit 2; }
	@$(UV_RUN) python .github/scripts/plan-local-tests --base "$(BASE)"

test-changed: ## Run changed-path tests, defaulting BASE to origin/main.
	@$(UV_RUN) python .github/scripts/plan-local-tests --base "$(or $(BASE),origin/main)" --execute

define run_topology_lane
	PYTEST_ADDOPTS="$(PYTEST_DIAGNOSTIC_ARGS) $(PYTEST_ARGS)" \
		$(TOPOLOGY_RUNNER) $(1) $(if $(TESTS),$(TESTS))
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
	$(UV_RUN) pytest -n 0 --timeout=30 --timeout-method=thread tests/unit/tooling/test_ci_compatibility.py $(PYTEST_DIAGNOSTIC_ARGS) $(PYTEST_ARGS)

test-affected: ## Execute planner-selected exact nodes or their fail-closed lanes.
	@test -n "$(BASE)" || { echo "BASE is required (for example: make test-affected BASE=origin/main)" >&2; exit 2; }
	@$(UV_RUN) python .github/scripts/plan-local-tests --base "$(BASE)" --execute

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
	$(UV_RUN) pytest -n 0 --timeout=120 --timeout-method=thread -m property --count=$(STRESS_COUNT) \
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

check-static: lint-full typecheck test-architecture todo-check build ## Run CI-owned static checks plus a local package build.

harbor-check: ## Verify committed Harbor task digests against local task contents.
	$(HARBOR_PYTHON) tools/check_harbor_dataset.py

harbor-oracle: harbor-check ## Run the Harbor Oracle contract gate.
	$(HARBOR_RUNNER) run -c benchmarks/regression-v1/job-oracle.json $(EVAL_ARGS)

agent-eval: ## Run the Harbor-native Jacobian workflow observation job.
	@if [ "$(EVAL_EXECUTE)" != "1" ]; then \
		echo "Model execution is opt-in. Review the job, then run: make agent-eval EVAL_EXECUTE=1"; \
		exit 0; \
	fi; \
	image=$${JACOBIAN_IMAGE:-}; \
	if ! printf '%s\n' "$$image" | grep -Eq '^.+@sha256:[0-9a-f]{64}$$'; then \
		echo "JACOBIAN_IMAGE must be an image reference pinned by @sha256:<64 lowercase hex digits>" >&2; \
		exit 2; \
	fi; \
	if [ -z "$${JACOBIAN_MCP_TOKEN:-}" ] || [ -z "$${JACOBIAN_AUTH_TOKENS_JSON:-}" ] || [ -z "$${JACOBIAN_MODEL:-}" ]; then \
		echo "JACOBIAN_MCP_TOKEN, JACOBIAN_AUTH_TOKENS_JSON, and JACOBIAN_MODEL must be exported" >&2; \
		exit 2; \
	fi; \
	$(MAKE) harbor-check && \
	resolved_job=$$(mktemp "$${TMPDIR:-/tmp}/jacobian-job.XXXXXX.json") && \
	trap 'rm -f "$$resolved_job"' EXIT HUP INT TERM && \
	$(UV_RUN) python tools/render_harbor_job.py \
		--input benchmarks/regression-v1/job-jacobian.json \
		--output "$$resolved_job" \
		--model "$${JACOBIAN_MODEL}" && \
	$(HARBOR_RUNNER) run -c "$$resolved_job" $(EVAL_ARGS)

bench-core: ## Run the core performance benchmark script.
	$(UV_RUN) python benchmarks/benchmark_core.py

clean: ## Remove local caches, build outputs, and coverage artifacts.
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build htmlcov
	rm -f .coverage .coverage.*
	find src tests benchmarks -type d -name '__pycache__' -prune -exec rm -rf {} +
	find . -maxdepth 2 -type d -name '*.egg-info' -prune -exec rm -rf {} +

docs-linkcheck: ## Check relative Markdown links in project docs.
	npx --yes markdown-link-check@3.15.0 \
		--config .markdown-link-check.json -q \
		README.md AGENTS.md CONTRIBUTING.md docs
