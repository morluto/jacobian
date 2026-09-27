from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict, cast
from urllib.parse import urlsplit

import yaml

ROOT = Path(__file__).parents[2]


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


class _JobEnvironment(TypedDict, total=False):
    extra_allowed_hosts: list[str]
    extra_docker_compose: list[str]


class _Dataset(TypedDict):
    path: str
    task_names: list[str]


class _Artifact(TypedDict):
    source: str
    service: str


class _HarborJob(TypedDict, total=False):
    agents: list[dict[str, object]]
    artifacts: list[str | _Artifact]
    datasets: list[_Dataset]
    environment: _JobEnvironment
    n_attempts: int


class _ComposeService(TypedDict, total=False):
    command: list[str]
    depends_on: dict[str, dict[str, str]]
    environment: dict[str, str]
    extra_hosts: list[str]
    healthcheck: dict[str, object]
    volumes: list[dict[str, object]]


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _read_harbor_job(path: Path) -> _HarborJob:
    return cast(_HarborJob, _read_json(path))


def _read_yaml(path: Path) -> dict[str, object]:
    value: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def _compose_services(path: Path) -> dict[str, _ComposeService]:
    parsed = _read_yaml(path)
    services = parsed.get("services")
    assert isinstance(services, dict)
    return cast(dict[str, _ComposeService], services)


def test_agent_eval_docs_exclude_host_codex_from_the_control_protocol() -> None:
    guide = (ROOT / "benchmarks" / "docs" / "run-agent-evaluations.md").read_text(
        encoding="utf-8"
    )

    assert "fresh temporary `CODEX_HOME`" in guide
    assert "direct host `codex exec`" in guide
    assert "control must have no Jacobian MCP server" in guide
    assert "treatment must" in guide
    assert "no Jacobian Skill" in guide
    assert "Build images through an opt-in proxy builder" in guide
    assert "JACOBIAN_EVAL_BUILDX_BUILDER=jacobian-eval-proxy" in guide
    assert "Docker daemon proxy" in guide


def test_proxy_observation_job_is_opt_in_and_preserves_local_mcp_access() -> None:
    proxy_job = _read_harbor_job(OBSERVATION_PROXY_JOB)
    proxy_control = _read_harbor_job(CONTROL_PROXY_JOB)
    assert proxy_job["environment"]["extra_docker_compose"] == [
        "benchmarks/config/agent-eval-egress-proxy.compose.yaml",
        "benchmarks/datasets/mathematical-benchmarks-v1/jacobian-observation.compose.yaml",
    ]
    assert proxy_job["artifacts"] == [
        "/logs/agent/trajectory.json",
        {"source": "/logs/jacobian/mcp.log", "service": "jacobian"},
    ]
    assert proxy_control["artifacts"] == ["/logs/agent/trajectory.json"]


def test_proxy_control_job_is_valid_harbor_job_json() -> None:
    job = _read_harbor_job(CONTROL_PROXY_JOB)

    assert job["n_attempts"] == 3
    assert job["datasets"] == [
        {
            "path": "benchmarks/datasets/mathematical-benchmarks-v1",
            "task_names": ["graph-counterexample"],
        }
    ]
    assert job["environment"]["extra_docker_compose"] == [
        "benchmarks/config/agent-eval-egress-proxy.compose.yaml",
    ]


def test_proxy_compose_overlay_declares_proxy_environment() -> None:
    services = _compose_services(EGRESS_PROXY_COMPOSE)
    main = services["main"]

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
    assert env["NO_PROXY"] == "${JACOBIAN_EVAL_NO_PROXY:-localhost,127.0.0.1,jacobian}"
    assert env["no_proxy"] == env["NO_PROXY"]

    sidecar = services["harbor-docker-egress-control-sidecar"]
    assert "host.docker.internal:host-gateway" in sidecar["extra_hosts"]
    assert sidecar["volumes"] == [
        {
            "type": "bind",
            "source": "${JACOBIAN_EVAL_GOST_CONFIG:?set JACOBIAN_EVAL_GOST_CONFIG to the rendered GOST configuration}",
            "target": "/opt/egress-sidecar/gost.yaml",
            "read_only": True,
        }
    ]


def test_observation_compose_overlay_declares_jacobian_service() -> None:
    services = _compose_services(OBSERVATION_COMPOSE)
    jacobian = services["jacobian"]
    command = jacobian["command"]
    main = services["main"]

    assert 'exec uv run --no-sync jacobian-remote-mcp "$@"' in command[0]
    assert command[1] == "jacobian-remote-mcp"
    assert "--transport" in command
    assert "--allow-anonymous" in command
    assert 'exec uv run --no-sync jacobian-mcp "$@"' not in command[0]
    assert main["depends_on"]["jacobian"]["condition"] == "service_healthy"
    healthcheck = jacobian["healthcheck"]
    check = healthcheck["test"]
    assert isinstance(check, list)
    assert isinstance(check[3], str)
    port = command[command.index("--port") + 1]
    assert "socket.create_connection" in check[3]
    assert f"'127.0.0.1', {port})" in check[3]
    assert port == "8000"
    config = _read_json(ROOT / "benchmarks" / "config" / "jacobian.mcp.json")
    mcp_servers = config["mcp_servers"]
    assert isinstance(mcp_servers, list)
    server = mcp_servers[0]
    assert isinstance(server, dict)
    url = server["url"]
    assert isinstance(url, str)
    assert urlsplit(url).port == int(port)
    assert "networks" not in jacobian
    assert "--state-dir" not in command
    assert not jacobian.get("volumes")


def test_paired_jobs_keep_the_same_egress_allowlist() -> None:
    treatment = _read_harbor_job(JOB)
    control = _read_harbor_job(CONTROL_JOB)

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
