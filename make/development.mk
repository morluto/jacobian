.PHONY: uv-version-check setup doctor setup-agent container-image eval-image eval-image-pull eval-image-bind deploy-check hooks fix lint complexity-check lint-full security-audit typecheck test-architecture test-runtime-inventory compile-test-plan compile-test-plan-check architecture docs-command-check docs-linkcheck

PROFILE ?= core

uv-version-check: ## Require the repository-pinned uv release.
	@test "$$(uv --version | awk '{print $$2}')" = "$$(tr -d '[:space:]' < .uv-version)" || { echo "install uv $$(tr -d '[:space:]' < .uv-version) before using this checkout" >&2; exit 2; }

setup: ## Install and diagnose a locked development profile (PROFILE=core).
	python3 tools/development_profiles.py setup --profile "$(PROFILE)" --repo .

doctor: ## Diagnose a development profile without changing it (PROFILE=core).
	uv run --locked --no-sync python tools/development_profiles.py doctor --profile "$(PROFILE)" --repo .

setup-agent: ## Configure an agent against this source checkout (ARGS="--client codex --profile full-python").
	./scripts/setup-agent $(ARGS)

JACOBIAN_REGISTRY_IMAGE ?= ghcr.io/morluto/jacobian

container-image: ## Build jacobian:local from the current tree, including dirty changes.
	$(UV_RUN) python -m tools.manage_jacobian_image build --image "$(or $(IMAGE),jacobian:local)"

eval-image: ## Select a published digest for a clean tree or build jacobian:local when dirty.
	$(UV_RUN) python -m tools.manage_jacobian_image select --registry-image "$(JACOBIAN_REGISTRY_IMAGE)"

eval-image-pull: ## Pull the current clean revision and print its digest-pinned image reference.
	$(UV_RUN) python -m tools.manage_jacobian_image pull --registry-image "$(JACOBIAN_REGISTRY_IMAGE)"

eval-image-bind: ## Bind image identity into RUNTIME_SNAPSHOT (JACOBIAN_IMAGE=..., RUNTIME_SNAPSHOT=...).
	@test -n "$(JACOBIAN_IMAGE)" -a -n "$(RUNTIME_SNAPSHOT)" || { echo "JACOBIAN_IMAGE and RUNTIME_SNAPSHOT are required" >&2; exit 2; }
	$(UV_RUN) python -m tools.manage_jacobian_image bind-runtime --image "$(JACOBIAN_IMAGE)" --runtime-snapshot "$(RUNTIME_SNAPSHOT)"

deploy-check: ## Validate the clone-to-systemd deployment entrypoint.
	bash -n deploy/install.sh
	$(PYTEST_RUNNER) --name deploy-check -- -n 0 tests/boundary/process/tooling/test_deploy_installer.py

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
	$(UV_RUN) python -m tools.check_test_architecture .

test-runtime-inventory: ## Fail when authorized complete-runtime uses lack verify/authority signals.
	$(UV_RUN) python -m tools.inventory_test_runtime --fail-on-unjustified

compile-test-plan: ## Regenerate topology.toml and ci-impact.json from plan_manifest.toml.
	$(UV_RUN) python -m tools.test_plan.compile --write

compile-test-plan-check: ## Fail when topology or impact projections are stale.
	$(UV_RUN) python -m tools.test_plan.compile --check

architecture: ## Enforce product source boundary invariants (subprocess, shutil.which, environ, contracts, surfaces).
	$(UV_RUN) python tools/check_architecture.py

docs-command-check: ## Validate Make targets and TESTS paths in command examples.
	$(UV_RUN) python tools/check_doc_commands.py

docs-linkcheck: docs-command-check ## Check relative Markdown links in project docs.
	npx --yes markdown-link-check@3.15.0 --config .markdown-link-check.json -q README.md README.zh-CN.md AGENTS.md CONTRIBUTING.md docs
