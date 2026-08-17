# Harbor-only. Product CI does not classify or plan from PATHS.
# Changed-path temps live only inside recipes (EXIT trap); never at parse time.

HARBOR_VERSION ?= 0.20.0
HARBOR_RUNNER ?= uvx --from harbor==$(HARBOR_VERSION) harbor
HARBOR_PYTHON ?= uvx --from harbor==$(HARBOR_VERSION) --with tomli-w==1.2.0 --with jsonschema python
HARBOR_PROJECT_PYTHON ?= uv run --locked --with harbor==$(HARBOR_VERSION) --with tomli-w==1.2.0 --with jsonschema python
# Validation pytest is process-isolated under xdist; Path monkeypatches stay
# worker-local. Oracle/adapter Make targets remain serial.
HARBOR_VALIDATION_WORKERS ?= 2
HARBOR_VALIDATION_TOTAL_WORKERS ?= 4
HARBOR_ORACLE_LOCK ?= benchmarks/results/.harbor-oracle.lock
HARBOR_ORACLE_DOCKER_BUILD_MODE ?= auto

# Compose Bake can reuse the wrong same-named Dockerfile with the BuildKit
# bundled by pre-23 Docker engines. Keep modern BuildKit, but fail over to the
# classic builder on older daemons and bind the resolved runtime into evidence.
define _resolve_harbor_oracle_docker_build_mode
docker_server_version="$$(docker version --format '{{.Server.Version}}')"; \
docker_compose_version="$$(docker compose version --short)"; \
docker_build_mode="$(HARBOR_ORACLE_DOCKER_BUILD_MODE)"; \
if [ "$$docker_build_mode" = "auto" ]; then \
	docker_server_major="$${docker_server_version%%.*}"; \
	case "$$docker_server_major" in ''|*[!0-9]*) echo "unable to parse Docker server version: $$docker_server_version" >&2; exit 2;; esac; \
	if [ "$$docker_server_major" -lt 23 ]; then docker_build_mode="legacy"; else docker_build_mode="buildkit"; fi; \
fi; \
case "$$docker_build_mode" in \
	legacy) export DOCKER_BUILDKIT=0 COMPOSE_BAKE=false;; \
	buildkit) export DOCKER_BUILDKIT=1 COMPOSE_BAKE=true;; \
	*) echo "HARBOR_ORACLE_DOCKER_BUILD_MODE must be auto, legacy, or buildkit" >&2; exit 2;; \
esac;
endef

.PHONY: harbor-plan harbor-prepare-task harbor-validate-task harbor-sync harbor-contracts harbor-execution-check harbor-adapter-checks harbor-validation-tests harbor-host-validation harbor-validate harbor-check harbor-check-all harbor-check-task benchmark-inventory benchmark-snapshot benchmark-snapshot-validate benchmark-publish harbor-oracle harbor-oracle-task harbor-oracle-run harbor-oracle-all harbor-adapter-check heldout-validate heldout-render heldout-smoke

harbor-plan: export JACOBIAN_HARBOR_PATHS := $(PATHS)
harbor-plan: ## Write one canonical Harbor plan.json for the current changes.
	@set -eu; \
	tmp_dir=$$(mktemp -d); \
	trap 'rm -rf "$$tmp_dir"' EXIT; \
	base_sha=""; \
	base_arg=""; \
	if [ -n "$(BASE)" ]; then \
		base_sha=$$(git rev-parse "$(BASE)"); \
		base_arg="--base $$base_sha"; \
	fi; \
	head_sha=$$(git rev-parse HEAD); \
	if [ -n "$$JACOBIAN_HARBOR_PATHS" ]; then \
		printf '%s\n' "$$JACOBIAN_HARBOR_PATHS" > "$$tmp_dir/raw-paths.txt"; \
	else \
		{ \
			if [ -n "$(BASE)" ]; then git diff --name-only "$(BASE)" HEAD; fi; \
			git diff --name-only HEAD; \
			git ls-files --others --exclude-standard; \
		} | sort -u > "$$tmp_dir/raw-paths.txt"; \
	fi; \
	$(UV_RUN) python .github/scripts/normalize-ci-paths --file "$$tmp_dir/raw-paths.txt" > "$$tmp_dir/changed-paths.txt"; \
	$(HARBOR_PYTHON) .github/scripts/plan-benchmarks \
		$$base_arg --head "$$head_sha" --paths-file "$$tmp_dir/changed-paths.txt" \
		--output "$$tmp_dir/plan.json"; \
	echo "Benchmark plan:"; \
	cat "$$tmp_dir/plan.json"

harbor-prepare-task: ## Format and sync selected Harbor tasks (DATASET=..., TASKS="...").
	@test -n "$(DATASET)" || { echo "DATASET is required" >&2; exit 2; }
	@test -n "$(TASKS)" || { echo "TASKS is required; refusing an unscoped preparation" >&2; exit 2; }
	$(HARBOR_PYTHON) tools/harbor_task_workflow.py prepare \
		--dataset "$(DATASET)" --tasks $(TASKS)

harbor-validate-task: ## Run the complete selected-task static, host, and Oracle gate.
	@test -n "$(DATASET)" || { echo "DATASET is required" >&2; exit 2; }
	@test -n "$(TASKS)" || { echo "TASKS is required; refusing an implicit full-dataset validation" >&2; exit 2; }
	$(HARBOR_PYTHON) tools/harbor_task_workflow.py validate \
		--dataset "$(DATASET)" --tasks $(TASKS)

harbor-sync: ## Update verifier checksum labels for selected tasks (DATASET=... TASKS="...").
	@test -n "$(DATASET)" || { echo "DATASET is required" >&2; exit 2; }
	@test -n "$(TASKS)" || { echo "TASKS is required; refusing an unscoped checksum update" >&2; exit 2; }
	$(HARBOR_PYTHON) -m benchmarks.tooling.public_contract sync-dataset \
		--dataset-root "benchmarks/datasets/$(DATASET)" --tasks $(TASKS)
	$(HARBOR_PYTHON) tools/sync_harbor_verifier_support.py \
		--dataset "$(DATASET)" --tasks $(TASKS)

harbor-contracts: ## Check Harbor sync, task topology, schemas, and generated records.
	$(HARBOR_PYTHON) tools/check_harbor_dataset.py --check
	$(HARBOR_PYTHON) tools/check_benchmark_contracts.py

harbor-execution-check: harbor-contracts ## Check Harbor jobs, MCP config, Compose, and execution helpers.
	$(PYTEST_RUNNER) --name harbor-execution -- -n 0 tests/tooling/test_harbor*.py \
		$(PYTEST_DIAGNOSTIC_ARGS) $(PYTEST_ARGS)

harbor-adapter-checks: ## Check every repository-owned Harbor adapter.
	@set -eu; for adapter_dir in benchmarks/adapters/*; do \
		test -d "$$adapter_dir" || continue; \
		$(MAKE) --no-print-directory harbor-adapter-check \
			ADAPTER="$${adapter_dir##*/}"; \
	done

harbor-validation-tests: ## Run Harbor's host-side validation test suite.
	$(PYTEST_RUNNER) --name "$(or $(PYTEST_RUN_NAME),harbor-validation)" -- -n $(HARBOR_VALIDATION_WORKERS) \
		$(PYTEST_DIAGNOSTIC_ARGS) $(if $(TESTS),$(TESTS),benchmarks/validation) $(PYTEST_ARGS)

harbor-host-validation: ## Run the full host suite in timing-balanced local shards.
	$(VALIDATION_LOCK) run --target harbor-host-validation -- $(MAKE) _harbor-host-validation

_harbor-host-validation:
	$(UV_RUN) python -m benchmarks.tooling.host_validation run-full \
		--total-workers $(HARBOR_VALIDATION_TOTAL_WORKERS) --max-parallel 4

harbor-check: harbor-execution-check harbor-adapter-checks ## Check Harbor contracts and control-plane behavior.

harbor-check-all: ## Explicitly run every repository-owned Harbor host regression.
	$(VALIDATION_LOCK) run --target harbor-check-all -- $(MAKE) _harbor-check-all

_harbor-check-all: harbor-check _harbor-host-validation

harbor-validate: harbor-check-all ## Backward-compatible exhaustive Harbor validation alias.

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
	@mkdir -p "$(dir $(HARBOR_ORACLE_LOCK))"
	@exec 9>"$(HARBOR_ORACLE_LOCK)"; flock 9; \
	$(_resolve_harbor_oracle_docker_build_mode) \
	job_name="$$( $(HARBOR_PYTHON) benchmarks/tooling/validate_harbor_results.py \
		--prepare --dataset "$(DATASET)" \
		--jobs-dir "benchmarks/results/$(DATASET)-oracle" \
		--job-config "benchmarks/datasets/$(DATASET)/jobs/oracle.json" \
		--execution-args="$(EVAL_ARGS)" \
		--docker-build-mode "$$docker_build_mode" \
		--docker-server-version "$$docker_server_version" \
		--docker-compose-version "$$docker_compose_version" \
		--tasks $(TASKS) )" && \
	$(HARBOR_RUNNER) run \
		-c "benchmarks/datasets/$(DATASET)/jobs/oracle.json" \
		-p "benchmarks/datasets/$(DATASET)" \
		$(foreach task,$(TASKS),--include-task-name "$(task)") \
		$(EVAL_ARGS) \
		--job-name "$$job_name" && \
	$(HARBOR_PYTHON) benchmarks/tooling/validate_harbor_results.py \
		--dataset "$(DATASET)" \
		--jobs-dir "benchmarks/results/$(DATASET)-oracle" \
		--job-config "benchmarks/datasets/$(DATASET)/jobs/oracle.json" \
		--execution-args="$(EVAL_ARGS)" \
		--docker-build-mode "$$docker_build_mode" \
		--docker-server-version "$$docker_server_version" \
		--docker-compose-version "$$docker_compose_version" \
		--result "benchmarks/results/$(DATASET)-oracle/$$job_name/result.json" \
		--tasks $(TASKS)

harbor-oracle-run: ## Run a dataset Oracle after an already-successful contract gate.
	@test -n "$(DATASET)" || { echo "DATASET is required" >&2; exit 2; }
	@test -n "$(TASKS)" -o "$(FULL)" = "1" || { echo "TASKS is required; use FULL=1 only for an intentional full-dataset Oracle" >&2; exit 2; }
	@test -f "benchmarks/datasets/$(DATASET)/jobs/oracle.json" || { echo "unknown dataset or missing Oracle job: $(DATASET)" >&2; exit 2; }
	@mkdir -p "$(dir $(HARBOR_ORACLE_LOCK))"
	@exec 9>"$(HARBOR_ORACLE_LOCK)"; flock 9; \
	$(_resolve_harbor_oracle_docker_build_mode) \
	job_name="$$( $(HARBOR_PYTHON) benchmarks/tooling/validate_harbor_results.py \
		--prepare --dataset "$(DATASET)" \
		--jobs-dir "benchmarks/results/$(DATASET)-oracle" \
		--job-config "benchmarks/datasets/$(DATASET)/jobs/oracle.json" \
		--execution-args="$(EVAL_ARGS)" \
		--docker-build-mode "$$docker_build_mode" \
		--docker-server-version "$$docker_server_version" \
		--docker-compose-version "$$docker_compose_version" \
		$(if $(TASKS),--tasks $(TASKS),) )" && \
	$(HARBOR_RUNNER) run \
		-c "benchmarks/datasets/$(DATASET)/jobs/oracle.json" \
		-p "benchmarks/datasets/$(DATASET)" \
		$(foreach task,$(TASKS),--include-task-name "$(task)") \
		$(EVAL_ARGS) \
		--job-name "$$job_name" && \
	$(HARBOR_PYTHON) benchmarks/tooling/validate_harbor_results.py \
		--dataset "$(DATASET)" \
		--jobs-dir "benchmarks/results/$(DATASET)-oracle" \
		--job-config "benchmarks/datasets/$(DATASET)/jobs/oracle.json" \
		--execution-args="$(EVAL_ARGS)" \
		--docker-build-mode "$$docker_build_mode" \
		--docker-server-version "$$docker_server_version" \
		--docker-compose-version "$$docker_compose_version" \
		--result "benchmarks/results/$(DATASET)-oracle/$$job_name/result.json" \
		$(if $(TASKS),--tasks $(TASKS),)

harbor-oracle-all: ## Run every registered dataset Oracle with tasks.
	$(VALIDATION_LOCK) run --target harbor-oracle-all -- $(MAKE) _harbor-oracle-all

_harbor-oracle-all: _harbor-check-all
	@set -e; for dataset in mathematical-benchmarks-v1 symbolic-coordination-v1 public-reproductions-v1 conjecture-probes-v1; do \
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
