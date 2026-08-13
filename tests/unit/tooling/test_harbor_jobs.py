from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from benchmarks.tooling.benchmark_contracts import (
    benchmark_contract_inventory,
    collect_contract_failures,
    validate_job_contract,
)
from benchmarks.tooling.harbor_suite import load_registry

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
