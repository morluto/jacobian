from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.tooling.benchmark_contracts import validate_all

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
    assert job["agents"] == [{"name": "codex", "kwargs": {"web_search": "disabled"}}]


@pytest.mark.timeout(30)
def test_validate_all_covers_proxy_control_and_observation_jobs() -> None:
    """The execution-config gate must validate proxied job configs, not skip them.

    ``validate_all`` is the contract layer beneath ``make harbor-contracts``,
    which is the first step of ``make harbor-execution-check``.  A malformed
    proxied control or observation job must be caught here rather than only
    failing when an operator runs the proxied evaluation.
    """
    failures = validate_all()

    # The committed proxy jobs are valid, so there must be no failures.
    assert failures == [], "benchmark contract failures:\n" + "\n".join(failures)


@pytest.mark.timeout(30)
def test_validate_all_catches_a_malformed_proxy_control_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed proxied control job must not pass the execution-config gate."""
    from benchmarks.tooling import benchmark_contracts

    original_read_json = benchmark_contracts._read_json

    def patched_read_json(path: Path) -> object:
        if path.name == "mathematical-benchmarks-v1-control-proxy.json":
            return {"n_attempts": "not-an-integer"}
        return original_read_json(path)

    monkeypatch.setattr(benchmark_contracts, "_read_json", patched_read_json)

    failures = validate_all()

    assert any("control-proxy" in f for f in failures), (
        f"expected a contract failure for the malformed proxy control job, "
        f"got: {failures}"
    )


def test_paired_jobs_use_three_attempts_per_condition() -> None:
    treatment = _read_json(JOB)
    control = _read_json(CONTROL_JOB)

    assert treatment["n_attempts"] == 3
    assert control["n_attempts"] == 3
