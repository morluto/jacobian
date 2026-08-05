from __future__ import annotations

import json
from pathlib import Path

from benchmarks.tooling.harbor_suite import get_suite

ROOT = Path(__file__).parents[3]
JOB = (
    ROOT
    / "benchmarks"
    / "datasets"
    / "agent-workflow-v1"
    / "jobs"
    / "jacobian-observation.json"
)
CONTROL_JOB = ROOT / "benchmarks" / "config" / "agent-workflow-v1-control.json"
MCP_CONFIG = ROOT / "benchmarks" / "config" / "jacobian.mcp.json"
LOOPBACK_MCP_CONFIG = ROOT / "benchmarks" / "config" / "jacobian-loopback.mcp.json"


def test_observation_job_uses_harbor_dataset_selection() -> None:
    job = json.loads(JOB.read_text())

    assert "tasks" not in job
    assert job["datasets"] == [
        {
            "path": "benchmarks/datasets/agent-workflow-v1",
            "task_names": ["graph-counterexample"],
        }
    ]
    assert job["agents"] == [{"name": "codex", "kwargs": {"web_search": "disabled"}}]
    assert job["environment"]["extra_allowed_hosts"] == [
        "api.openai.com",
        "auth.openai.com",
        "chatgpt.com",
        "deb.debian.org",
        "nodejs.org",
        "npmjs.org",
        "registry.npmjs.org",
        "raw.githubusercontent.com",
    ]


def test_observation_job_keeps_the_minimal_jacobian_treatment() -> None:
    job = json.loads(JOB.read_text())

    assert job["agents"] == [{"name": "codex", "kwargs": {"web_search": "disabled"}}]
    assert job["environment"]["extra_docker_compose"] == [
        "benchmarks/datasets/agent-workflow-v1/jacobian-observation.compose.yaml",
    ]


def test_agent_eval_forwards_web_search_setting_to_harbor() -> None:
    makefile = (ROOT / "Makefile").read_text()

    assert '--ak "web_search=$(CODEX_WEB_SEARCH)"' in makefile
    assert "JACOBIAN_EVAL_PROXY" in makefile
    assert (
        "JACOBIAN_EVAL_HTTP_PROXY ?= $(call _jacobian_eval_container_proxy,$(HTTP_PROXY))"
        in makefile
    )
    assert (
        "JACOBIAN_EVAL_HTTPS_PROXY ?= $(call _jacobian_eval_container_proxy,$(HTTPS_PROXY))"
        in makefile
    )
    assert (
        "JACOBIAN_EVAL_ALL_PROXY ?= $(call _jacobian_eval_container_proxy,$(ALL_PROXY))"
        in makefile
    )
    assert "JACOBIAN_EVAL_CODEX_BINARY" in makefile
    assert "JACOBIAN_EVAL_UPSTREAM_PROXY" in makefile
    assert "benchmarks.tooling.harbor_proxy" in makefile
    assert 'if [ "$(JACOBIAN_EVAL_PROXY)" = "1" ]; then' in makefile
    assert 'JACOBIAN_EVAL_NO_PROXY="$(JACOBIAN_EVAL_NO_PROXY)"' in makefile
    assert "agent-workflow-v1-control-proxy.json" in makefile
    assert "jacobian-observation-proxy.json" in makefile
    assert "jacobian-loopback.mcp.json" in makefile


def test_proxy_observation_job_is_opt_in_and_preserves_local_mcp_access() -> None:
    proxy_job = json.loads(
        (
            ROOT
            / "benchmarks"
            / "datasets"
            / "agent-workflow-v1"
            / "jobs"
            / "jacobian-observation-proxy.json"
        ).read_text()
    )
    proxy_overlay = (
        ROOT / "benchmarks" / "config" / "agent-eval-egress-proxy.compose.yaml"
    ).read_text()
    codex_overlay = (
        ROOT / "benchmarks" / "config" / "agent-eval-codex.compose.yaml"
    ).read_text()
    assert proxy_job["environment"]["extra_docker_compose"] == [
        "benchmarks/config/agent-eval-codex.compose.yaml",
        "benchmarks/config/agent-eval-egress-proxy.compose.yaml",
        "benchmarks/datasets/agent-workflow-v1/jacobian-observation.compose.yaml",
    ]
    assert "NO_PROXY" in proxy_overlay
    assert "127.0.0.1" in proxy_overlay
    assert "jacobian" in proxy_overlay
    assert "host.docker.internal:host-gateway" in proxy_overlay
    assert "harbor-docker-egress-control-sidecar:" in proxy_overlay
    assert "JACOBIAN_EVAL_GOST_CONFIG" in proxy_overlay
    assert 'HTTPS_PROXY: ""' in proxy_overlay
    assert "JACOBIAN_EVAL_CODEX_BINARY" in codex_overlay
    assert "target: /usr/local/bin/codex" in codex_overlay


def test_jacobian_sidecar_keeps_its_project_network_under_egress_control() -> None:
    observation_overlay = (
        ROOT
        / "benchmarks"
        / "datasets"
        / "agent-workflow-v1"
        / "jacobian-observation.compose.yaml"
    ).read_text()

    assert "jacobian:" in observation_overlay
    assert "networks:" not in observation_overlay
    assert "condition: service_healthy" in observation_overlay
    assert "socket.create_connection" in observation_overlay


def test_observation_mcp_config_is_external_to_the_task_job() -> None:
    job = json.loads(JOB.read_text())
    control = json.loads(CONTROL_JOB.read_text())
    mcp = json.loads(MCP_CONFIG.read_text())
    loopback_mcp = json.loads(LOOPBACK_MCP_CONFIG.read_text())

    assert "mcp_servers" not in job["agents"][0]
    assert "mcp_servers" not in control["agents"][0]
    assert mcp["mcp_servers"] == [
        {
            "name": "jacobian",
            "transport": "streamable-http",
            "url": "http://jacobian:8000/mcp",
        }
    ]
    assert loopback_mcp["mcp_servers"] == [
        {
            "name": "jacobian",
            "transport": "streamable-http",
            "url": "http://127.0.0.1:8000/mcp",
        }
    ]


def test_observation_dataset_contains_the_canonical_task() -> None:
    suite = get_suite("agent-workflow-v1")

    assert any(ref.path.name == "graph-counterexample" for ref in suite.tasks)


def test_paired_jobs_use_three_attempts_per_condition() -> None:
    treatment = json.loads(JOB.read_text())
    control = json.loads(CONTROL_JOB.read_text())

    assert treatment["n_attempts"] == 3
    assert control["n_attempts"] == 3


def test_paired_jobs_keep_the_same_egress_allowlist() -> None:
    treatment = json.loads(JOB.read_text())
    control = json.loads(CONTROL_JOB.read_text())

    assert (
        treatment["environment"]["extra_allowed_hosts"]
        == control["environment"]["extra_allowed_hosts"]
    )
