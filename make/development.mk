.PHONY: uv-version-check setup doctor setup-lean doctor-lean doctor-external setup-agent container-image eval-image eval-image-pull eval-image-bind deploy-check hooks fix lint complexity-check lint-full security-audit typecheck test-architecture test-runtime-inventory architecture docs-command-check docs-linkcheck

uv-version-check: ## Require the repository-pinned uv release.
	@test "$$(uv --version | awk '{print $$2}')" = "$$(tr -d '[:space:]' < .uv-version)" || { echo "install uv $$(tr -d '[:space:]' < .uv-version) before using this checkout" >&2; exit 2; }

setup: ## Install the locked contributor environment and diagnose Python backends.
	python3 tools/development_profiles.py setup --repo .

doctor: ## Diagnose the locked contributor environment without changing it.
	uv run --locked --no-sync python tools/development_profiles.py doctor --repo .

setup-lean: ## Install the locked environment and the pinned Lean toolchain.
	python3 tools/development_profiles.py setup --profile lean --repo .

doctor-lean: ## Diagnose Lean/elan/lake without changing the checkout.
	uv run --locked --no-sync python tools/development_profiles.py doctor --profile lean --repo .

doctor-external: ## Diagnose optional SAT proof binaries (no downloads).
	uv run --locked --no-sync python tools/development_profiles.py doctor --profile external-proof --repo .

setup-agent: ## Configure an agent against this source checkout (ARGS="--client codex --profile core").
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
	$(UV_RUN) python -m tools.check_test_architecture .

import-contracts: ## Enforce declared package dependency direction.
	$(UV_RUN) lint-imports

test-runtime-inventory: ## Fail when authorized complete-runtime uses lack verify/authority signals.
	$(UV_RUN) python -m tools.inventory_test_runtime --fail-on-unjustified

architecture: ## Enforce product source boundary invariants (subprocess, shutil.which, environ, contracts, surfaces).
	$(UV_RUN) python tools/check_architecture.py

docs-command-check: ## Validate Make targets and TESTS paths in command examples.
	$(UV_RUN) python tools/check_doc_commands.py

docs-linkcheck: docs-command-check ## Check relative Markdown links in project docs.
	npx --yes markdown-link-check@3.15.0 --config .markdown-link-check.json -q README.md AGENTS.md CONTRIBUTING.md docs
