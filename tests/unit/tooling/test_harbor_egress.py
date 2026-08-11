from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[3]


JOB = (
    ROOT
    / "benchmarks"
    / "datasets"
    / "mathematical-benchmarks-v1"
    / "jobs"
    / "jacobian-observation.json"
)
CONTROL_JOB = ROOT / "benchmarks" / "config" / "mathematical-benchmarks-v1-control.json"
CONTROL_PROXY_JOB = (
    ROOT / "benchmarks" / "config" / "mathematical-benchmarks-v1-control-proxy.json"
)
OBSERVATION_PROXY_JOB = (
    ROOT
    / "benchmarks"
    / "datasets"
    / "mathematical-benchmarks-v1"
    / "jobs"
    / "jacobian-observation-proxy.json"
)
CODEX_COMPOSE = ROOT / "benchmarks" / "config" / "agent-eval-codex.compose.yaml"
EGRESS_PROXY_COMPOSE = (
    ROOT / "benchmarks" / "config" / "agent-eval-egress-proxy.compose.yaml"
)
OBSERVATION_COMPOSE = (
    ROOT
    / "benchmarks"
    / "datasets"
    / "mathematical-benchmarks-v1"
    / "jacobian-observation.compose.yaml"
)


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_observation_job_keeps_the_minimal_jacobian_treatment() -> None:
    job = _read_json(JOB)

    assert job["agents"] == [
        {
            "name": "codex",
            "skills": [".agents/skills/jacobian-math"],
            "kwargs": {"web_search": "disabled"},
        }
    ]
    assert job["environment"]["extra_docker_compose"] == [
        "benchmarks/datasets/mathematical-benchmarks-v1/jacobian-observation.compose.yaml",
    ]


def test_agent_eval_forwards_web_search_setting_to_harbor() -> None:
    evaluations = (ROOT / "make" / "evaluations.mk").read_text(encoding="utf-8")

    assert '--ak "web_search=$(CODEX_WEB_SEARCH)"' in evaluations
    assert "JACOBIAN_EVAL_PROXY" in evaluations
    assert (
        "JACOBIAN_EVAL_HTTP_PROXY ?= $(call _jacobian_eval_container_proxy,$(HTTP_PROXY))"
        in evaluations
    )
    assert (
        "JACOBIAN_EVAL_HTTPS_PROXY ?= $(call _jacobian_eval_container_proxy,$(HTTPS_PROXY))"
        in evaluations
    )
    assert (
        "JACOBIAN_EVAL_ALL_PROXY ?= $(call _jacobian_eval_container_proxy,$(ALL_PROXY))"
        in evaluations
    )
    assert "JACOBIAN_EVAL_CODEX_BINARY" in evaluations
    assert "benchmarks.tooling.codex_binary" in evaluations
    assert "JACOBIAN_EVAL_UPSTREAM_PROXY" in evaluations
    assert "export CODEX_FORCE_AUTH_JSON=1" in evaluations
    assert "benchmarks.tooling.harbor_proxy" in evaluations
    assert 'if [ "$(JACOBIAN_EVAL_PROXY)" = "1" ]; then' in evaluations
    assert 'JACOBIAN_EVAL_NO_PROXY="$(JACOBIAN_EVAL_NO_PROXY)"' in evaluations
    assert "mathematical-benchmarks-v1-control-proxy.json" in evaluations
    assert "jacobian-observation-proxy.json" in evaluations
    assert "jacobian-loopback.mcp.json" in evaluations
    assert "JACOBIAN_EVAL_SKILL ?= .agents/skills/jacobian-math" in evaluations
    assert '--skill "$(JACOBIAN_EVAL_SKILL)"' in evaluations
    validate_recipe = evaluations.split("agent-eval-validate:", maxsplit=1)[1].split(
        "\n\n", maxsplit=1
    )[0]
    assert "$(HARBOR_PROJECT_PYTHON) -m benchmarks.tooling.observation_results" in (
        validate_recipe
    )


def test_agent_eval_docs_exclude_host_codex_from_the_control_protocol() -> None:
    guide = (ROOT / "docs" / "how-to" / "run-agent-evaluations.md").read_text(
        encoding="utf-8"
    )

    assert "fresh temporary `CODEX_HOME`" in guide
    assert "direct host `codex exec`" in guide
    assert "control must have empty `skills` and `mcp_servers`" in guide


def test_proxy_observation_job_is_opt_in_and_preserves_local_mcp_access() -> None:
    proxy_job = _read_json(OBSERVATION_PROXY_JOB)
    proxy_control = _read_json(CONTROL_PROXY_JOB)
    proxy_overlay = EGRESS_PROXY_COMPOSE.read_text(encoding="utf-8")
    codex_overlay = CODEX_COMPOSE.read_text(encoding="utf-8")
    assert proxy_job["environment"]["extra_docker_compose"] == [
        "benchmarks/config/agent-eval-codex.compose.yaml",
        "benchmarks/config/agent-eval-egress-proxy.compose.yaml",
        "benchmarks/datasets/mathematical-benchmarks-v1/jacobian-observation.compose.yaml",
    ]
    assert "NO_PROXY" in proxy_overlay
    assert "127.0.0.1" in proxy_overlay
    assert "jacobian" in proxy_overlay
    assert "host.docker.internal:host-gateway" in proxy_overlay
    assert "harbor-docker-egress-control-sidecar:" in proxy_overlay
    assert "JACOBIAN_EVAL_GOST_CONFIG" in proxy_overlay
    assert "http://127.0.0.1:12346" in proxy_overlay
    assert "JACOBIAN_EVAL_CODEX_BINARY" in codex_overlay
    assert "target: /usr/local/bin/codex" in codex_overlay
    assert proxy_job["artifacts"] == [
        "/logs/agent/trajectory.json",
        {"source": "/logs/jacobian/mcp.log", "service": "jacobian"},
    ]
    assert proxy_control["artifacts"] == ["/logs/agent/trajectory.json"]


def test_jacobian_sidecar_keeps_its_project_network_under_egress_control() -> None:
    observation_overlay = (
        ROOT
        / "benchmarks"
        / "datasets"
        / "mathematical-benchmarks-v1"
        / "jacobian-observation.compose.yaml"
    ).read_text(encoding="utf-8")

    assert "jacobian:" in observation_overlay
    assert "networks:" not in observation_overlay
    assert "condition: service_healthy" in observation_overlay
    assert "socket.create_connection" in observation_overlay


def test_proxy_control_job_is_valid_harbor_job_json() -> None:
    job = _read_json(CONTROL_PROXY_JOB)

    assert job["n_attempts"] == 3
    assert job["datasets"] == [
        {
            "path": "benchmarks/datasets/mathematical-benchmarks-v1",
            "task_names": ["graph-counterexample"],
        }
    ]
    assert job["environment"]["extra_docker_compose"] == [
        "benchmarks/config/agent-eval-codex.compose.yaml",
        "benchmarks/config/agent-eval-egress-proxy.compose.yaml",
    ]


def test_compose_overlays_parse_as_valid_yaml() -> None:
    """Syntactically broken overlays must not pass the execution-config gate.

    The gate claims to cover Compose overlays, so the owning tests must parse
    them as YAML rather than only asserting on substring presence.
    """
    for compose_path in (CODEX_COMPOSE, EGRESS_PROXY_COMPOSE, OBSERVATION_COMPOSE):
        parsed = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
        assert isinstance(parsed, dict), f"{compose_path.name}: root must be a mapping"
        assert "services" in parsed, f"{compose_path.name}: missing services table"
        assert isinstance(parsed["services"], dict), (
            f"{compose_path.name}: services must be a mapping"
        )


def test_proxy_compose_overlay_declares_proxy_environment() -> None:
    parsed = yaml.safe_load(EGRESS_PROXY_COMPOSE.read_text(encoding="utf-8"))
    main = parsed["services"]["main"]

    assert "environment" in main
    env = main["environment"]
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
    ):
        assert key in env, f"proxy compose missing {key}"
    assert env["HTTP_PROXY"] == "http://127.0.0.1:12346"
    assert env["HTTPS_PROXY"] == "http://127.0.0.1:12346"
    assert env["http_proxy"] == "http://127.0.0.1:12346"
    assert env["https_proxy"] == "http://127.0.0.1:12346"

    sidecar = parsed["services"]["harbor-docker-egress-control-sidecar"]
    assert "host.docker.internal:host-gateway" in sidecar["extra_hosts"]


def test_observation_compose_overlay_declares_jacobian_service() -> None:
    parsed = yaml.safe_load(OBSERVATION_COMPOSE.read_text(encoding="utf-8"))

    assert "jacobian" in parsed["services"]


def test_paired_jobs_keep_the_same_egress_allowlist() -> None:
    treatment = _read_json(JOB)
    control = _read_json(CONTROL_JOB)

    assert treatment["environment"]["extra_allowed_hosts"] == [
        "api.openai.com",
        "auth.openai.com",
        "chatgpt.com",
        "deb.debian.org",
        "nodejs.org",
        "npmjs.org",
        "registry.npmjs.org",
        "raw.githubusercontent.com",
    ]
    assert (
        treatment["environment"]["extra_allowed_hosts"]
        == control["environment"]["extra_allowed_hosts"]
    )
