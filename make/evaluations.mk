.PHONY: agent-eval agent-eval-validate agent-eval-compare codex-visibility codex-tool-context provider-eval

JACOBIAN_ENABLED ?= 1
ifneq ($(filter 0 1,$(JACOBIAN_ENABLED)),$(JACOBIAN_ENABLED))
$(error JACOBIAN_ENABLED must be exactly 0 or 1 (got '$(JACOBIAN_ENABLED)'))
endif

CODEX_WEB_SEARCH ?= disabled
JACOBIAN_EVAL_PROXY ?= 0
JACOBIAN_EVAL_CODEX_BINARY ?= $(shell command -v codex 2>/dev/null)
define _jacobian_eval_container_proxy
$(subst localhost,host.docker.internal,$(subst 127.0.0.1,host.docker.internal,$1))
endef
JACOBIAN_EVAL_HTTP_PROXY ?= $(call _jacobian_eval_container_proxy,$(HTTP_PROXY))
JACOBIAN_EVAL_HTTPS_PROXY ?= $(call _jacobian_eval_container_proxy,$(HTTPS_PROXY))
JACOBIAN_EVAL_ALL_PROXY ?= $(call _jacobian_eval_container_proxy,$(ALL_PROXY))
JACOBIAN_EVAL_NO_PROXY ?= localhost,127.0.0.1,jacobian
JACOBIAN_EVAL_UPSTREAM_PROXY ?= $(or $(JACOBIAN_EVAL_HTTPS_PROXY),$(JACOBIAN_EVAL_ALL_PROXY),$(JACOBIAN_EVAL_HTTP_PROXY))
JACOBIAN_EVAL_GOST_CONFIG ?= $(abspath benchmarks/results/.runtime/agent-eval-gost.yaml)
ifneq ($(filter 0 1,$(JACOBIAN_EVAL_PROXY)),$(JACOBIAN_EVAL_PROXY))
$(error JACOBIAN_EVAL_PROXY must be exactly 0 or 1 (got '$(JACOBIAN_EVAL_PROXY)'))
endif

ifeq ($(JACOBIAN_EVAL_PROXY),1)
ifneq ($(strip $(JACOBIAN_EVAL_HTTP_PROXY)$(JACOBIAN_EVAL_HTTPS_PROXY)$(JACOBIAN_EVAL_ALL_PROXY)),)
else
$(error JACOBIAN_EVAL_PROXY=1 requires JACOBIAN_EVAL_HTTP_PROXY, JACOBIAN_EVAL_HTTPS_PROXY, or JACOBIAN_EVAL_ALL_PROXY)
endif
endif

ifeq ($(JACOBIAN_ENABLED),0)
ifeq ($(JACOBIAN_EVAL_PROXY),1)
EVAL_CONFIG ?= benchmarks/config/mathematical-benchmarks-v1-control-proxy.json
else
EVAL_CONFIG ?= benchmarks/config/mathematical-benchmarks-v1-control.json
endif
override MCP_CONFIG :=
else
ifeq ($(JACOBIAN_EVAL_PROXY),1)
EVAL_CONFIG ?= benchmarks/datasets/$(or $(DATASET),mathematical-benchmarks-v1)/jobs/jacobian-observation-proxy.json
MCP_CONFIG ?= benchmarks/config/jacobian-loopback.mcp.json
else
EVAL_CONFIG ?= benchmarks/datasets/$(or $(DATASET),mathematical-benchmarks-v1)/jobs/jacobian-observation.json
MCP_CONFIG ?= benchmarks/config/jacobian.mcp.json
endif
endif

agent-eval: ## Run a Harbor evaluation (JACOBIAN_ENABLED=0|1, JACOBIAN_EVAL_PROXY=0|1, DATASET=mathematical-benchmarks-v1, EVAL_EXECUTE=1).
	@set -e; \
	CODEX_BINARY="$(JACOBIAN_EVAL_CODEX_BINARY)"; \
	if [ "$(EVAL_EXECUTE)" != "1" ]; then \
		echo "Model execution is opt-in. Review the job, then run: make agent-eval DATASET=mathematical-benchmarks-v1 EVAL_EXECUTE=1"; \
		exit 0; \
	fi; \
	if [ -z "$${JACOBIAN_MODEL:-}" ]; then \
		echo "JACOBIAN_MODEL must be exported" >&2; \
		exit 2; \
	fi; \
	if [ "$(JACOBIAN_EVAL_PROXY)" = "1" ]; then \
		CODEX_BINARY="$$( $(UV_RUN) python -m benchmarks.tooling.codex_binary --candidate "$$CODEX_BINARY" )"; \
		JACOBIAN_EVAL_UPSTREAM_PROXY="$(JACOBIAN_EVAL_UPSTREAM_PROXY)" \
			$(UV_RUN) python -m benchmarks.tooling.harbor_proxy \
			--output "$(JACOBIAN_EVAL_GOST_CONFIG)"; \
	fi; \
	if [ -z "$${OPENAI_API_KEY:-}" ] && [ -z "$${CODEX_FORCE_AUTH_JSON+x}" ] && [ -s "$${HOME}/.codex/auth.json" ]; then \
		export CODEX_FORCE_AUTH_JSON=1; \
	fi; \
	if [ "$(JACOBIAN_ENABLED)" = "1" ]; then \
		test -n "$${JACOBIAN_IMAGE:-}" || { echo "JACOBIAN_IMAGE must be exported" >&2; exit 2; }; \
		test -n "$(RUNTIME_SNAPSHOT)" || { echo "RUNTIME_SNAPSHOT is required for a Jacobian-enabled run" >&2; exit 2; }; \
		$(UV_RUN) python -m tools.manage_jacobian_image bind-runtime \
			--image "$${JACOBIAN_IMAGE}" --runtime-snapshot "$(RUNTIME_SNAPSHOT)"; \
	fi; \
	JACOBIAN_EVAL_HTTP_PROXY="$(JACOBIAN_EVAL_HTTP_PROXY)" \
	JACOBIAN_EVAL_HTTPS_PROXY="$(JACOBIAN_EVAL_HTTPS_PROXY)" \
	JACOBIAN_EVAL_ALL_PROXY="$(JACOBIAN_EVAL_ALL_PROXY)" \
	JACOBIAN_EVAL_NO_PROXY="$(JACOBIAN_EVAL_NO_PROXY)" \
	JACOBIAN_EVAL_CODEX_BINARY="$$CODEX_BINARY" \
	JACOBIAN_EVAL_GOST_CONFIG="$(JACOBIAN_EVAL_GOST_CONFIG)" \
	$(HARBOR_RUNNER) run \
		-c "$(EVAL_CONFIG)" \
		-a codex \
		-m "$${JACOBIAN_MODEL}" \
		--ak "web_search=$(CODEX_WEB_SEARCH)" \
		$(if $(MCP_CONFIG),--mcp-config "$(MCP_CONFIG)",) \
		$(if $(TASKS),-p "benchmarks/datasets/$(or $(DATASET),mathematical-benchmarks-v1)" $(foreach task,$(TASKS),--include-task-name "$(task)"),) \
		$(EVAL_ARGS)

agent-eval-validate: ## Normalize one observation (RESULTS=..., JOB=..., CONDITION=..., OUTPUT=...).
	@test -n "$(RESULTS)" -a -n "$(JOB)" -a -n "$(CONDITION)" -a -n "$(OUTPUT)" || { echo "RESULTS, JOB, CONDITION, and OUTPUT are required" >&2; exit 2; }
	$(HARBOR_PROJECT_PYTHON) -m benchmarks.tooling.observation_results validate \
		--dataset "$(or $(DATASET),mathematical-benchmarks-v1)" --condition "$(CONDITION)" \
		--job "$(JOB)" --jobs-dir "$(RESULTS)" --output "$(OUTPUT)" \
		$(if $(RESULT),--result "$(RESULT)",) \
		$(if $(RUNTIME_SNAPSHOT),--runtime-snapshot "$(RUNTIME_SNAPSHOT)",) \
		$(if $(HELDOUT_MANIFEST),--heldout-manifest "$(HELDOUT_MANIFEST)",)

agent-eval-compare: ## Compare normalized observations (CONTROL=..., TREATMENT=..., OUTPUT=...).
	@test -n "$(CONTROL)" -a -n "$(TREATMENT)" -a -n "$(OUTPUT)" || { echo "CONTROL, TREATMENT, and OUTPUT are required" >&2; exit 2; }
	$(UV_RUN) python -m benchmarks.tooling.observation_results compare \
		--control "$(CONTROL)" --treatment "$(TREATMENT)" --output "$(OUTPUT)"

VISIBILITY_CASES ?= benchmarks/config/codex-visibility-v2.json
VISIBILITY_REPETITIONS ?= 1
VISIBILITY_REASONING_EFFORT ?= high
VISIBILITY_TOOL_MODE ?= direct

codex-visibility: ## Measure Codex adoption of Jacobian (VISIBILITY_EXECUTE=1, VISIBILITY_MCP_URL=..., VISIBILITY_MODEL=..., VISIBILITY_OUTPUT=...).
	@set -e; \
	if [ "$(VISIBILITY_EXECUTE)" != "1" ]; then \
		echo "Model execution is opt-in. Set VISIBILITY_EXECUTE=1 after reviewing $(VISIBILITY_CASES)."; \
		exit 0; \
	fi; \
	test -n "$(VISIBILITY_MCP_URL)" -a -n "$(VISIBILITY_MODEL)" -a -n "$(VISIBILITY_OUTPUT)" || { \
		echo "VISIBILITY_MCP_URL, VISIBILITY_MODEL, and VISIBILITY_OUTPUT are required" >&2; \
		exit 2; \
	}; \
	$(UV_RUN) python -m benchmarks.tooling.codex_visibility \
		--execute --cases "$(VISIBILITY_CASES)" --mcp-url "$(VISIBILITY_MCP_URL)" \
		--model "$(VISIBILITY_MODEL)" --reasoning-effort "$(VISIBILITY_REASONING_EFFORT)" \
		--tool-mode "$(VISIBILITY_TOOL_MODE)" \
		--repetitions "$(VISIBILITY_REPETITIONS)" --output "$(VISIBILITY_OUTPUT)" \
		$(foreach case,$(VISIBILITY_CASES_SELECTED),--case "$(case)")

codex-tool-context: ## Measure ALL_TOOLS projection cost in Codex ATIF traces (TRAJECTORIES="...").
	@test -n "$(TRAJECTORIES)" || { echo "TRAJECTORIES is required" >&2; exit 2; }
	$(UV_RUN) python -m benchmarks.tooling.codex_tool_context $(TRAJECTORIES) \
		$(if $(LABEL),--label "$(LABEL)",) $(if $(OUTPUT),--output "$(OUTPUT)",)

provider-eval: ## Run pinned provider feasibility jobs (PROVIDER=cddlib|cgal|gudhi|lean-repl|nauty|regina).
	@test -n "$(PROVIDER)" || { echo "PROVIDER is required" >&2; exit 2; }
	@case "$(PROVIDER)" in cddlib|cgal|gudhi|lean-repl|nauty|regina) ;; *) echo "unknown provider: $(PROVIDER)" >&2; exit 2;; esac
	@$(MAKE) harbor-check && \
	$(HARBOR_RUNNER) run \
		-c benchmarks/datasets/provider-feasibility-v1/jobs/oracle.json \
		-p benchmarks/datasets/provider-feasibility-v1 \
		--include-task-name "$(PROVIDER)" \
		$(EVAL_ARGS)
