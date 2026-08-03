.DEFAULT_GOAL := help

UV_RUN := uv run --locked
HARBOR_VERSION ?= 0.20.0
HARBOR_RUNNER ?= uvx --from harbor==$(HARBOR_VERSION) harbor
HARBOR_PYTHON ?= uvx --from harbor==$(HARBOR_VERSION) --with tomli-w==1.2.0 --with jsonschema python
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
.PHONY: help uv-version-check setup setup-agent container-image hooks fix lint complexity-check lint-full security-audit typecheck test-architecture test-plan test-changed test-unit test-component test-domain test-composition test-storage test-process test-mcp test-provider test-lean test-e2e test-affected test-all-ci test-compatibility test-stress test-ordering duplicate-code npm-test todo-check coverage build check precommit check-static harbor-plan harbor-sync harbor-validate harbor-check harbor-check-task benchmark-inventory benchmark-snapshot benchmark-snapshot-validate benchmark-publish harbor-oracle harbor-oracle-task harbor-oracle-run harbor-oracle-all harbor-adapter-check heldout-validate heldout-render heldout-smoke agent-eval agent-eval-validate agent-eval-compare provider-eval clean docs-linkcheck deploy-check

help: ## Show available developer commands.
	@awk 'BEGIN {FS = ":.*## "; printf "Jacobian developer commands:\n\n"} /^[a-zA-Z_-]+:.*## / {printf "  %-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

uv-version-check: ## Require the repository-pinned uv release.
	@test "$$(uv --version | awk '{print $$2}')" = "$$(tr -d '[:space:]' < .uv-version)" || { \
		echo "install uv $$(tr -d '[:space:]' < .uv-version) before using this checkout" >&2; \
		exit 2; \
	}

setup: uv-version-check ## Install the locked development environment.
	uv sync --locked --dev

setup-agent: ## Configure an agent against this source checkout (ARGS="--client codex --profile full-python").
	./scripts/setup-agent $(ARGS)

container-image: ## Build a revision-labelled local image (IMAGE=jacobian:local).
	@test -z "$$(git status --porcelain)" || { echo "container-image requires a clean worktree" >&2; exit 2; }
	@revision=$$(git rev-parse HEAD); \
	version=$$(uv version --short); \
	docker build \
		--build-arg JACOBIAN_REVISION="$$revision" \
		--build-arg JACOBIAN_VERSION="$$version" \
		-t "$(or $(IMAGE),jacobian:local)" .

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

harbor-plan: ## Print the independent Harbor benchmark plan (BASE=... optional).
	@changed_paths=$$({ \
		if [ -n "$(BASE)" ]; then git diff --name-only "$(BASE)" HEAD; fi; \
		git diff --name-only HEAD; \
		git ls-files --others --exclude-standard; \
	} | sort -u); \
	$(HARBOR_PYTHON) .github/scripts/plan-benchmarks $(if $(BASE),--base "$(BASE)",) -- $$changed_paths

harbor-sync: ## Update vendored verifier support and deterministic task digests.
	$(HARBOR_PYTHON) tools/sync_harbor_verifier_support.py --write
	$(HARBOR_PYTHON) tools/check_harbor_dataset.py --write

harbor-validate: ## Run all repository-owned Harbor checks under the pinned Harbor runtime.
	$(HARBOR_PYTHON) tools/sync_harbor_verifier_support.py --check
	$(HARBOR_PYTHON) tools/check_harbor_dataset.py --check
	$(HARBOR_PYTHON) tools/check_benchmark_contracts.py
	@set -eu; for adapter_dir in benchmarks/adapters/*; do \
		test -d "$$adapter_dir" || continue; \
		$(MAKE) --no-print-directory harbor-adapter-check \
			ADAPTER="$${adapter_dir##*/}"; \
	done
	$(UV_RUN) pytest -n 0 benchmarks/validation

harbor-check: harbor-validate ## Run Harbor topology, digest, provenance, and host-side validation checks.

harbor-check-task: ## Validate selected Harbor leaf tasks (DATASET=..., TASKS="task-a task-b").
	@test -n "$(DATASET)" || { echo "DATASET is required" >&2; exit 2; }
	@test -n "$(TASKS)" || { echo "TASKS is required; refusing an implicit full-dataset check" >&2; exit 2; }
	$(HARBOR_PYTHON) tools/check_harbor_dataset.py \
		--dataset "$(DATASET)" --tasks $(TASKS)

benchmark-inventory: ## Render the content-bound benchmark inventory (OUTPUT=path optional).
	$(HARBOR_PYTHON) -m benchmarks.tooling.benchmark_inventory $(if $(OUTPUT),--output "$(OUTPUT)",)

benchmark-snapshot: ## Create an immutable snapshot lock (DATASET=..., OUTPUT=..., SOURCE_TREE=... optional).
	@test -n "$(DATASET)" || { echo "DATASET is required" >&2; exit 2; }
	$(HARBOR_PYTHON) tools/manage_benchmark_snapshots.py create \
		--dataset "$(DATASET)" $(if $(OUTPUT),--output "$(OUTPUT)",) \
		$(if $(SOURCE_TREE),--source-tree "$(SOURCE_TREE)",)

benchmark-snapshot-validate: ## Validate a committed snapshot lock (LOCK=..., REPRODUCE=1 optional).
	@test -n "$(LOCK)" || { echo "LOCK is required" >&2; exit 2; }
	$(HARBOR_PYTHON) tools/manage_benchmark_snapshots.py validate --lock "$(LOCK)" \
		$(if $(filter 1,$(REPRODUCE)),--reproduce,) \
		$(if $(SOURCE_TREE),--source-tree "$(SOURCE_TREE)",)

benchmark-publish: ## Generate an ignored Harbor dataset.toml from a snapshot (LOCK=..., DEST=... optional).
	@test -n "$(LOCK)" || { echo "LOCK is required" >&2; exit 2; }
	$(HARBOR_PYTHON) tools/manage_benchmark_snapshots.py publish --lock "$(LOCK)" \
		$(if $(DEST),--dest "$(DEST)",)

harbor-oracle: ## Check contracts, then run an explicitly scoped dataset Oracle.
	@test -n "$(DATASET)" || { echo "DATASET is required" >&2; exit 2; }
	@test -n "$(TASKS)" -o "$(FULL)" = "1" || { echo "TASKS is required; use FULL=1 only for an intentional full-dataset Oracle" >&2; exit 2; }
	$(MAKE) harbor-check
	$(MAKE) harbor-oracle-run DATASET="$(DATASET)" TASKS="$(TASKS)" FULL="$(FULL)" EVAL_ARGS="$(EVAL_ARGS)"

harbor-oracle-task: harbor-check-task ## Check selected leaf tasks, then run their exact Oracle.
	@test -f "benchmarks/datasets/$(DATASET)/jobs/oracle.json" || { echo "unknown dataset or missing Oracle job: $(DATASET)" >&2; exit 2; }
	$(HARBOR_RUNNER) run \
		-c "benchmarks/datasets/$(DATASET)/jobs/oracle.json" \
		-p "benchmarks/datasets/$(DATASET)" \
		$(foreach task,$(TASKS),--include-task-name "$(task)") \
		$(EVAL_ARGS) && \
	$(HARBOR_PYTHON) benchmarks/tooling/validate_harbor_results.py \
		--dataset "$(DATASET)" \
		--jobs-dir "benchmarks/results/$(DATASET)-oracle" \
		--tasks $(TASKS)

harbor-oracle-run: ## Run a dataset Oracle after an already-successful contract gate.
	@test -n "$(DATASET)" || { echo "DATASET is required" >&2; exit 2; }
	@test -n "$(TASKS)" -o "$(FULL)" = "1" || { echo "TASKS is required; use FULL=1 only for an intentional full-dataset Oracle" >&2; exit 2; }
	@test -f "benchmarks/datasets/$(DATASET)/jobs/oracle.json" || { echo "unknown dataset or missing Oracle job: $(DATASET)" >&2; exit 2; }
	$(HARBOR_RUNNER) run \
		-c "benchmarks/datasets/$(DATASET)/jobs/oracle.json" \
		-p "benchmarks/datasets/$(DATASET)" \
		$(foreach task,$(TASKS),--include-task-name "$(task)") \
		$(EVAL_ARGS) && \
	$(HARBOR_PYTHON) benchmarks/tooling/validate_harbor_results.py \
		--dataset "$(DATASET)" \
		--jobs-dir "benchmarks/results/$(DATASET)-oracle" \
		$(if $(TASKS),--tasks $(TASKS),)

harbor-oracle-all: harbor-check ## Run every registered dataset Oracle with tasks.
	@set -e; for dataset in agent-workflow-v1 public-reproductions-v1 research-diagnostics-v1 provider-feasibility-v1; do \
		$(MAKE) --no-print-directory harbor-oracle-run DATASET=$$dataset FULL=1 EVAL_ARGS="$(EVAL_ARGS)"; \
	done

harbor-adapter-check: ## Check deterministic regeneration for ADAPTER=<id>.
	@test -n "$(ADAPTER)" || { echo "ADAPTER is required" >&2; exit 2; }
	$(HARBOR_PYTHON) tools/check_benchmark_adapters.py --adapter "$(ADAPTER)"
	@test -x "benchmarks/adapters/$(ADAPTER)/check.sh" || { echo "adapter check.sh is missing: $(ADAPTER)" >&2; exit 2; }
	HARBOR_PYTHON="$(HARBOR_PYTHON)" "benchmarks/adapters/$(ADAPTER)/check.sh"

heldout-validate: ## Validate a private held-out manifest (MANIFEST=path).
	@test -n "$(MANIFEST)" || { echo "MANIFEST is required" >&2; exit 2; }
	$(UV_RUN) python -m benchmarks.tooling.heldout_bundle validate --manifest "$(MANIFEST)"

heldout-render: ## Verify a private bundle and render matched jobs (MANIFEST=..., BUNDLE_ROOT=..., OUTPUT=..., STAGE=...).
	@test -n "$(MANIFEST)" -a -n "$(BUNDLE_ROOT)" -a -n "$(OUTPUT)" -a -n "$(STAGE)" -a -n "$(MAX_TOKENS)" -a -n "$(MAX_COST_USD)" || { echo "MANIFEST, BUNDLE_ROOT, OUTPUT, STAGE, MAX_TOKENS, and MAX_COST_USD are required" >&2; exit 2; }
	$(HARBOR_PYTHON) -m benchmarks.tooling.heldout_bundle render \
		--manifest "$(MANIFEST)" --bundle-root "$(BUNDLE_ROOT)" --output "$(OUTPUT)" \
		--stage "$(STAGE)" --max-tokens "$(MAX_TOKENS)" --max-cost-usd "$(MAX_COST_USD)"

heldout-smoke: ## Render a synthetic private bundle and run Harbor nop/Oracle without model cost.
	$(HARBOR_PYTHON) -m benchmarks.tooling.heldout_smoke --run-harbor

JACOBIAN_ENABLED ?= 1
ifneq ($(filter 0 1,$(JACOBIAN_ENABLED)),$(JACOBIAN_ENABLED))
$(error JACOBIAN_ENABLED must be exactly 0 or 1 (got '$(JACOBIAN_ENABLED)'))
endif

ifeq ($(JACOBIAN_ENABLED),0)
EVAL_CONFIG ?= benchmarks/config/agent-workflow-v1-control.json
override MCP_CONFIG :=
else
EVAL_CONFIG ?= benchmarks/datasets/$(or $(DATASET),agent-workflow-v1)/jobs/jacobian-observation.json
MCP_CONFIG ?= benchmarks/config/jacobian.mcp.json
endif

agent-eval: ## Run a Harbor evaluation (JACOBIAN_ENABLED=0|1, DATASET=agent-workflow-v1, EVAL_EXECUTE=1).
	@if [ "$(EVAL_EXECUTE)" != "1" ]; then \
		echo "Model execution is opt-in. Review the job, then run: make agent-eval DATASET=agent-workflow-v1 EVAL_EXECUTE=1"; \
		exit 0; \
	fi; \
	if [ -z "$${JACOBIAN_MODEL:-}" ]; then \
		echo "JACOBIAN_MODEL must be exported" >&2; \
		exit 2; \
	fi; \
	$(HARBOR_RUNNER) run \
		-c "$(EVAL_CONFIG)" \
		-a codex \
		-m "$${JACOBIAN_MODEL}" \
		$(if $(MCP_CONFIG),--mcp-config "$(MCP_CONFIG)",) \
		$(if $(TASKS),-p "benchmarks/datasets/$(or $(DATASET),agent-workflow-v1)" $(foreach task,$(TASKS),--include-task-name "$(task)"),) \
		$(EVAL_ARGS)

agent-eval-validate: ## Normalize one observation (RESULTS=..., JOB=..., CONDITION=..., OUTPUT=...).
	@test -n "$(RESULTS)" -a -n "$(JOB)" -a -n "$(CONDITION)" -a -n "$(OUTPUT)" || { echo "RESULTS, JOB, CONDITION, and OUTPUT are required" >&2; exit 2; }
	$(UV_RUN) python -m benchmarks.tooling.observation_results validate \
		--dataset "$(or $(DATASET),agent-workflow-v1)" --condition "$(CONDITION)" \
		--job "$(JOB)" --jobs-dir "$(RESULTS)" --output "$(OUTPUT)" \
		$(if $(RESULT),--result "$(RESULT)",) \
		$(if $(RUNTIME_SNAPSHOT),--runtime-snapshot "$(RUNTIME_SNAPSHOT)",) \
		$(if $(HELDOUT_MANIFEST),--heldout-manifest "$(HELDOUT_MANIFEST)",)

agent-eval-compare: ## Compare normalized observations (CONTROL=..., TREATMENT=..., OUTPUT=...).
	@test -n "$(CONTROL)" -a -n "$(TREATMENT)" -a -n "$(OUTPUT)" || { echo "CONTROL, TREATMENT, and OUTPUT are required" >&2; exit 2; }
	$(UV_RUN) python -m benchmarks.tooling.observation_results compare \
		--control "$(CONTROL)" --treatment "$(TREATMENT)" --output "$(OUTPUT)"

provider-eval: ## Run pinned provider feasibility jobs (PROVIDER=cddlib|cgal|gudhi|lean-repl|nauty|regina).
	@test -n "$(PROVIDER)" || { echo "PROVIDER is required" >&2; exit 2; }
	@case "$(PROVIDER)" in cddlib|cgal|gudhi|lean-repl|nauty|regina) ;; *) echo "unknown provider: $(PROVIDER)" >&2; exit 2;; esac
	@$(MAKE) harbor-check && \
	$(HARBOR_RUNNER) run \
		-c benchmarks/datasets/provider-feasibility-v1/jobs/oracle.json \
		-p benchmarks/datasets/provider-feasibility-v1 \
		--include-task-name "$(PROVIDER)" \
		$(EVAL_ARGS)

clean: ## Remove local caches, build outputs, and coverage artifacts.
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build htmlcov
	rm -f .coverage .coverage.*
	find src tests benchmarks -type d -name '__pycache__' -prune -exec rm -rf {} +
	find . -maxdepth 2 -type d -name '*.egg-info' -prune -exec rm -rf {} +

docs-linkcheck: ## Check relative Markdown links in project docs.
	npx --yes markdown-link-check@3.15.0 \
		--config .markdown-link-check.json -q \
		README.md AGENTS.md CONTRIBUTING.md docs
