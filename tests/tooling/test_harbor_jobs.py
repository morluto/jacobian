from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

import pytest
from benchmarks.tooling.benchmark_contracts import (
    benchmark_contract_inventory,
    collect_contract_failures,
    validate_job_contract,
)
from benchmarks.tooling.command_runner import ToolCommandStatus, run_operator_command
from benchmarks.tooling.harbor_suite import load_registry

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


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_observation_job_uses_harbor_dataset_selection() -> None:
    job = _read_json(JOB)

    assert "tasks" not in job
    assert job["datasets"] == [
        {
            "path": "benchmarks/datasets/mathematical-benchmarks-v1",
            "task_names": ["graph-counterexample"],
        }
    ]
    assert job["agents"] == [
        {
            "name": "codex",
            "kwargs": {"web_search": "disabled"},
        }
    ]


def test_benchmark_inventory_covers_proxy_control_and_observation_jobs() -> None:
    """The execution-config gate must validate proxied job configs, not skip them.

    The inventory is consumed by ``validate_all``, the contract layer beneath
    ``make harbor-contracts``. A missing proxy entry would therefore remove it
    from the repository gate.
    """
    inventory = benchmark_contract_inventory()

    assert tuple(path.name for path in inventory.proxy_jobs) == (
        "mathematical-benchmarks-v1-control-proxy.json",
        "jacobian-observation-proxy.json",
    )


def test_job_contract_rejects_a_malformed_proxy_control_job() -> None:
    """A malformed proxied control job must not pass the execution-config gate."""
    path = benchmark_contract_inventory().proxy_jobs[0]
    malformed = _read_json(path)
    malformed["artifacts"] = ["logs/agent/trajectory.json"]

    failures = validate_job_contract(
        malformed,
        path=path,
        suite=load_registry()[0],
    )

    assert any("control-proxy" in f and "artifacts" in f for f in failures), (
        f"expected a contract failure for the malformed proxy control job, "
        f"got: {failures}"
    )


def test_contract_failure_collection_runs_every_phase_in_order() -> None:
    calls: list[str] = []

    def phase(name: str, *failures: str) -> Callable[[], list[str]]:
        def validate() -> list[str]:
            calls.append(name)
            return list(failures)

        return validate

    failures = collect_contract_failures(
        (
            phase("schemas", "schema failure"),
            phase("proxy jobs"),
            phase("snapshots", "snapshot failure 1", "snapshot failure 2"),
        )
    )

    assert calls == ["schemas", "proxy jobs", "snapshots"]
    assert failures == [
        "schema failure",
        "snapshot failure 1",
        "snapshot failure 2",
    ]


def test_paired_jobs_use_three_attempts_per_condition() -> None:
    treatment = _read_json(JOB)
    control = _read_json(CONTROL_JOB)

    assert treatment["n_attempts"] == 3
    assert control["n_attempts"] == 3


def test_paired_jobs_collect_runtime_evidence_available_in_each_condition() -> None:
    treatment = _read_json(JOB)
    control = _read_json(CONTROL_JOB)

    assert treatment["artifacts"] == [
        "/logs/agent/trajectory.json",
        {"source": "/logs/jacobian/mcp.log", "service": "jacobian"},
    ]
    assert control["artifacts"] == ["/logs/agent/trajectory.json"]


def test_agent_eval_resolves_a_current_image_when_not_explicitly_set(
    tmp_path: Path,
) -> None:
    """The treatment defaults to the clean revision's immutable registry image."""
    trace = tmp_path / "trace.txt"
    selected = "registry.invalid/jacobian@sha256:" + "a" * 64
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        "#!/bin/sh\n"
        'case "$*" in\n'
        "  *'tools.manage_jacobian_image select'*)\n"
        '    printf \'%s\\n\' "$*" >> "$TRACE"\n'
        "    printf '%s\\n' \"$SELECTED_IMAGE\"\n"
        "    ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    fake_harbor = tmp_path / "harbor"
    fake_harbor.write_text(
        '#!/bin/sh\nprintf \'harbor image=%s\\n\' "$JACOBIAN_IMAGE" >> "$TRACE"\n',
        encoding="utf-8",
    )
    fake_harbor.chmod(0o755)
    environment = os.environ | {
        "JACOBIAN_IMAGE": "",
        "SELECTED_IMAGE": selected,
        "TRACE": str(trace),
    }

    completed = run_operator_command(
        "make",
        (
            "agent-eval",
            "EVAL_EXECUTE=1",
            "JACOBIAN_MODEL=test-model",
            f"UV_RUN={fake_uv}",
            f"HARBOR_RUNNER={fake_harbor}",
            "JACOBIAN_REGISTRY_IMAGE=registry.invalid/jacobian",
        ),
        cwd=ROOT,
        environment=environment,
        timeout_seconds=120.0,
    )

    assert completed.status is ToolCommandStatus.EXITED
    assert completed.exit_code == 0, completed.stderr.decode("utf-8", errors="replace")
    assert trace.read_text(encoding="utf-8").splitlines() == [
        "python -m tools.manage_jacobian_image select --registry-image registry.invalid/jacobian",
        f"harbor image={selected}",
    ]


@pytest.mark.parametrize(
    ("proxy", "expected_job"),
    [
        ("0", "jacobian-observation.json"),
        ("1", "jacobian-observation-proxy.json"),
    ],
)
def test_agent_eval_keeps_the_local_mcp_endpoint_independent_of_egress_proxy(
    tmp_path: Path,
    proxy: str,
    expected_job: str,
) -> None:
    """Harbor egress control shares service networking, so MCP stays on loopback."""
    trace = tmp_path / "harbor-args.txt"
    fake_uv = tmp_path / "uv"
    fake_uv.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_uv.chmod(0o755)
    fake_harbor = tmp_path / "harbor"
    fake_harbor.write_text(
        '#!/bin/sh\nprintf \'%s\\n\' "$@" > "$TRACE"\n',
        encoding="utf-8",
    )
    fake_harbor.chmod(0o755)

    completed = run_operator_command(
        "make",
        (
            "agent-eval",
            "EVAL_EXECUTE=1",
            "JACOBIAN_MODEL=test-model",
            "JACOBIAN_IMAGE=jacobian:test",
            f"JACOBIAN_EVAL_PROXY={proxy}",
            "JACOBIAN_EVAL_HTTP_PROXY=http://proxy.invalid:7890",
            "EVAL_ATTEMPTS=2",
            "EVAL_REASONING_EFFORT=high",
            f"UV_RUN={fake_uv}",
            f"HARBOR_RUNNER={fake_harbor}",
        ),
        cwd=ROOT,
        environment=os.environ | {"TRACE": str(trace)},
        timeout_seconds=120.0,
    )

    assert completed.status is ToolCommandStatus.EXITED
    assert completed.exit_code == 0, completed.stderr.decode("utf-8", errors="replace")
    arguments = trace.read_text(encoding="utf-8").splitlines()
    assert arguments[arguments.index("-c") + 1].endswith(expected_job)
    mcp_index = arguments.index("--mcp-config")
    assert arguments[mcp_index + 1] == "benchmarks/config/jacobian-loopback.mcp.json"
    attempts_index = arguments.index("--n-attempts")
    assert arguments[attempts_index + 1] == "2"
    assert "reasoning_effort=high" in arguments


def test_agent_eval_forwards_an_explicit_buildx_builder(tmp_path: Path) -> None:
    trace = tmp_path / "buildx-builder.txt"
    fake_harbor = tmp_path / "harbor"
    fake_harbor.write_text(
        '#!/bin/sh\nprintf \'%s\\n\' "${BUILDX_BUILDER-unset}" > "$TRACE"\n',
        encoding="utf-8",
    )
    fake_harbor.chmod(0o755)

    completed = run_operator_command(
        "make",
        (
            "agent-eval",
            "EVAL_EXECUTE=1",
            "JACOBIAN_MODEL=test-model",
            "JACOBIAN_IMAGE=jacobian:test",
            "JACOBIAN_EVAL_BUILDX_BUILDER=jacobian-eval-proxy",
            f"HARBOR_RUNNER={fake_harbor}",
        ),
        cwd=ROOT,
        environment=os.environ | {"TRACE": str(trace)},
        timeout_seconds=120.0,
    )

    assert completed.status is ToolCommandStatus.EXITED
    assert completed.exit_code == 0, completed.stderr.decode("utf-8", errors="replace")
    assert trace.read_text(encoding="utf-8") == "jacobian-eval-proxy\n"


def test_agent_eval_creates_an_explicit_proxy_builder(tmp_path: Path) -> None:
    trace = tmp_path / "docker-args.txt"
    fake_docker = tmp_path / "docker"
    fake_docker.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" >> "$TRACE"\n'
        '[ "$1 $2" = "buildx inspect" ] && exit 1\n'
        '[ "$1 $2" = "buildx create" ] && exit 0\n'
        "exit 2\n",
        encoding="utf-8",
    )
    fake_docker.chmod(0o755)

    completed = run_operator_command(
        "make",
        (
            "agent-eval-proxy-builder-create",
            f"DOCKER={fake_docker}",
            "JACOBIAN_EVAL_BUILDX_PROXY=http://proxy.invalid:7890",
        ),
        cwd=ROOT,
        environment=os.environ | {"TRACE": str(trace)},
        timeout_seconds=120.0,
    )

    assert completed.status is ToolCommandStatus.EXITED
    assert completed.exit_code == 0, completed.stderr.decode("utf-8", errors="replace")
    assert trace.read_text(encoding="utf-8").splitlines() == [
        "buildx inspect jacobian-eval-proxy",
        "buildx create --name jacobian-eval-proxy --driver docker-container --driver-opt network=host --driver-opt env.HTTP_PROXY=http://proxy.invalid:7890 --driver-opt env.HTTPS_PROXY=http://proxy.invalid:7890 --driver-opt env.http_proxy=http://proxy.invalid:7890 --driver-opt env.https_proxy=http://proxy.invalid:7890 --bootstrap",
    ]


@pytest.mark.parametrize(
    ("configured_proxy", "expects_warning"),
    [
        ("", False),
        ("http://proxy.invalid:7890", True),
    ],
)
def test_agent_eval_reports_direct_egress_and_only_warns_for_a_configured_proxy(
    tmp_path: Path, configured_proxy: str, expects_warning: bool
) -> None:
    fake_harbor = tmp_path / "harbor"
    fake_harbor.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_harbor.chmod(0o755)

    completed = run_operator_command(
        "make",
        (
            "agent-eval",
            "EVAL_EXECUTE=1",
            "JACOBIAN_MODEL=test-model",
            "JACOBIAN_IMAGE=jacobian:test",
            "JACOBIAN_EVAL_PROXY=0",
            f"JACOBIAN_EVAL_HTTP_PROXY={configured_proxy}",
            f"JACOBIAN_EVAL_HTTPS_PROXY={configured_proxy}",
            f"JACOBIAN_EVAL_ALL_PROXY={configured_proxy}",
            f"HARBOR_RUNNER={fake_harbor}",
        ),
        cwd=ROOT,
        environment=os.environ,
        timeout_seconds=120.0,
    )

    assert completed.status is ToolCommandStatus.EXITED
    assert completed.exit_code == 0, completed.stderr.decode("utf-8", errors="replace")
    output = completed.stdout.decode("utf-8", errors="replace")
    assert "Provider egress: direct" in output
    warning = "Detected HTTP_PROXY / HTTPS_PROXY / ALL_PROXY on the host."
    hint = "If this network requires the proxy, rerun with JACOBIAN_EVAL_PROXY=1."
    assert (warning in output) is expects_warning
    assert (hint in output) is expects_warning
    assert "proxy.invalid" not in output
