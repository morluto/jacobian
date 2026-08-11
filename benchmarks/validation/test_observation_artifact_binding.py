"""Tests for observation evidence field binding and fail-closed behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.tooling import observation_results
from benchmarks.tooling.observation_results import (
    _jacobian_image_failures,
    _resolve_binding,
    _trial_status,
    build_observation_evidence,
)
from benchmarks.validation.observation_results_support import (
    _HARBOR_VERSION,
    _JACOBIAN_IMAGE,
    _SNAPSHOT_ID,
    _write_observation_job,
    _write_result,
)

# ---------------------------------------------------------------------------
# Normalization integration
# ---------------------------------------------------------------------------


def test_reproducible_treatment_requires_digest_bound_clean_image() -> None:
    runtime = {
        "condition": {"jacobian_enabled": True},
        "jacobian_image": {**_JACOBIAN_IMAGE, "source_dirty": True},
    }

    failures = _jacobian_image_failures(runtime)

    assert failures == ["Jacobian image must come from a clean source revision"]


def test_reproducible_treatment_requires_image_identity() -> None:
    failures = _jacobian_image_failures({"condition": {"jacobian_enabled": True}})

    assert failures == ["Jacobian-enabled runtime snapshot must bind jacobian_image"]


@pytest.mark.parametrize(
    "raw_status",
    ("RUNNING", "PENDING", "FAILED", "ERROR", "TIMEOUT", "CANCELLED"),
)
def test_trial_status_preserves_noncompleted_states(raw_status: str) -> None:
    assert _trial_status({"status": raw_status}, None) == raw_status


def test_trial_status_fails_closed_on_unknown_state() -> None:
    assert _trial_status({"status": "UNKNOWN"}, None) == "ERROR"


def test_observation_normalization_binds_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(observation_results, "task_digest", lambda _path: "a" * 64)
    monkeypatch.setattr(observation_results, "_git_sha", lambda: "b" * 40)
    job = {
        "jobs_dir": str(tmp_path / "jobs"),
        "n_attempts": 1,
        "timeout_multiplier": 1,
        "orchestrator": {"type": "local", "n_concurrent_trials": 1},
        "environment": {"type": "docker"},
        "agents": [{"name": "codex"}],
        "datasets": [
            {
                "path": "benchmarks/datasets/mathematical-benchmarks-v1",
                "task_names": ["graph-counterexample"],
            }
        ],
    }
    job_path = _write_observation_job(tmp_path, job)
    result_path = _write_result(tmp_path)

    evidence, failures = build_observation_evidence(
        dataset="mathematical-benchmarks-v1",
        condition="control",
        job_path=job_path,
        jobs_dir=tmp_path,
        result_path=result_path,
        runtime_snapshot={
            "snapshot_id": _SNAPSHOT_ID,
            "harbor_version": _HARBOR_VERSION,
            "model": "model",
            "condition": {
                "id": "control",
                "role": "PRIMARY_CONTROL",
                "jacobian_enabled": False,
            },
        },
    )

    assert failures == []
    assert evidence["status"] == "VALID"
    assert evidence["schema_version"] == "4"
    assert evidence["fixed_invariants"]["model"] == "model"
    assert evidence["eval_args"]["selection_mode"] == "dataset-task-names"
    assert evidence["eval_args"]["selection"] == ["graph-counterexample"]
    assert evidence["snapshot_id"] == _SNAPSHOT_ID
    assert evidence["harbor_version"] == _HARBOR_VERSION
    trial = evidence["trials"][0]
    assert trial["agent"] == {"name": "codex", "version": "1"}
    assert trial["verifier_state"] == "COMPLETED"
    assert trial["budgets"] == {"max_tokens": None, "max_cost_usd": None}
    assert trial["artifacts"] == []


def test_observation_binds_runtime_snapshot_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(observation_results, "task_digest", lambda _path: "a" * 64)
    monkeypatch.setattr(observation_results, "_git_sha", lambda: "b" * 40)
    job = {
        "jobs_dir": str(tmp_path / "jobs"),
        "n_attempts": 1,
        "timeout_multiplier": 1,
        "orchestrator": {"type": "local", "n_concurrent_trials": 1},
        "environment": {"type": "docker"},
        "agents": [{"name": "codex"}],
        "datasets": [
            {
                "path": "benchmarks/datasets/mathematical-benchmarks-v1",
                "task_names": ["graph-counterexample"],
            }
        ],
    }
    job_path = _write_observation_job(tmp_path, job)
    result_path = _write_result(tmp_path)
    runtime = {
        "snapshot_id": _SNAPSHOT_ID,
        "harbor_version": _HARBOR_VERSION,
        "model": "model",
        "agent": {"name": "codex", "version": "1"},
        "max_tokens": 100000,
        "max_cost_usd": 100.0,
        "repetition": 0,
        "pair_id": "graph-counterexample-r001",
        "jacobian_image": _JACOBIAN_IMAGE,
        "condition": {
            "id": "treatment",
            "role": "PRIMARY_TREATMENT",
            "jacobian_enabled": True,
        },
    }

    evidence, failures = build_observation_evidence(
        dataset="mathematical-benchmarks-v1",
        condition="treatment",
        job_path=job_path,
        jobs_dir=tmp_path,
        result_path=result_path,
        runtime_snapshot=runtime,
    )

    assert failures == []
    assert evidence["status"] == "VALID"
    assert evidence["snapshot_id"] == _SNAPSHOT_ID
    assert evidence["harbor_version"] == _HARBOR_VERSION
    trial = evidence["trials"][0]
    assert trial["budgets"] == {"max_tokens": 100000, "max_cost_usd": 100.0}
    assert trial["pair_id"] == "graph-counterexample-r001"
    assert evidence["fixed_invariants"]["runtime"]["harbor_version"] == _HARBOR_VERSION


# ---------------------------------------------------------------------------
# snapshot_id / harbor_version explicit bindings
# ---------------------------------------------------------------------------


def test_resolve_binding_agrees_on_single_source() -> None:
    value, failures = _resolve_binding(
        "snapshot_id",
        job={"snapshot_id": _SNAPSHOT_ID},
        runtime={},
    )

    assert failures == []
    assert value == _SNAPSHOT_ID


def test_resolve_binding_agrees_across_job_and_runtime() -> None:
    value, failures = _resolve_binding(
        "harbor_version",
        job={"harbor_version": _HARBOR_VERSION},
        runtime={"harbor_version": _HARBOR_VERSION},
        heldout_value=_HARBOR_VERSION,
    )

    assert failures == []
    assert value == _HARBOR_VERSION


def test_resolve_binding_rejects_mismatch() -> None:
    _value, failures = _resolve_binding(
        "harbor_version",
        job={"harbor_version": "0.20.0"},
        runtime={"harbor_version": "0.19.0"},
    )

    assert any("disagree" in f for f in failures)


def test_resolve_binding_rejects_missing() -> None:
    _value, failures = _resolve_binding("snapshot_id", job={}, runtime={})

    assert any("missing" in f for f in failures)


def test_observation_rejects_missing_snapshot_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(observation_results, "task_digest", lambda _path: "a" * 64)
    monkeypatch.setattr(observation_results, "_git_sha", lambda: "b" * 40)
    job = {
        "jobs_dir": str(tmp_path / "jobs"),
        "n_attempts": 1,
        "timeout_multiplier": 1,
        "orchestrator": {"type": "local", "n_concurrent_trials": 1},
        "environment": {"type": "docker"},
        "agents": [{"name": "codex"}],
        "datasets": [
            {
                "path": "benchmarks/datasets/mathematical-benchmarks-v1",
                "task_names": ["graph-counterexample"],
            }
        ],
    }
    job_path = _write_observation_job(
        tmp_path, job, snapshot_id=None, harbor_version=_HARBOR_VERSION
    )
    result_path = _write_result(tmp_path)

    evidence, failures = build_observation_evidence(
        dataset="mathematical-benchmarks-v1",
        condition="control",
        job_path=job_path,
        jobs_dir=tmp_path,
        result_path=result_path,
    )

    assert evidence["status"] == "INCOMPLETE"
    assert any("snapshot_id" in f and "missing" in f for f in failures)


def test_observation_rejects_mismatched_harbor_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(observation_results, "task_digest", lambda _path: "a" * 64)
    monkeypatch.setattr(observation_results, "_git_sha", lambda: "b" * 40)
    job = {
        "jobs_dir": str(tmp_path / "jobs"),
        "n_attempts": 1,
        "timeout_multiplier": 1,
        "orchestrator": {"type": "local", "n_concurrent_trials": 1},
        "environment": {"type": "docker"},
        "agents": [{"name": "codex"}],
        "datasets": [
            {
                "path": "benchmarks/datasets/mathematical-benchmarks-v1",
                "task_names": ["graph-counterexample"],
            }
        ],
    }
    job_path = _write_observation_job(
        tmp_path, job, snapshot_id=_SNAPSHOT_ID, harbor_version="0.19.0"
    )
    result_path = _write_result(tmp_path)
    runtime = {
        "snapshot_id": _SNAPSHOT_ID,
        "harbor_version": _HARBOR_VERSION,
    }

    evidence, failures = build_observation_evidence(
        dataset="mathematical-benchmarks-v1",
        condition="control",
        job_path=job_path,
        jobs_dir=tmp_path,
        result_path=result_path,
        runtime_snapshot=runtime,
    )

    assert evidence["status"] == "INCOMPLETE"
    assert any("harbor_version" in f and "disagree" in f for f in failures)


# ---------------------------------------------------------------------------
# Fail-closed behavior
# ---------------------------------------------------------------------------


def test_incomplete_execution_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(observation_results, "task_digest", lambda _path: "a" * 64)
    monkeypatch.setattr(observation_results, "_git_sha", lambda: "b" * 40)
    job = {
        "jobs_dir": str(tmp_path / "jobs"),
        "n_attempts": 1,
        "timeout_multiplier": 1,
        "orchestrator": {"type": "local", "n_concurrent_trials": 1},
        "environment": {"type": "docker"},
        "agents": [{"name": "codex"}],
        "datasets": [
            {
                "path": "benchmarks/datasets/mathematical-benchmarks-v1",
                "task_names": ["graph-counterexample"],
            }
        ],
    }
    job_path = _write_observation_job(tmp_path, job)
    result = _write_result(tmp_path)
    payload = json.loads(result.read_text(encoding="utf-8"))
    payload["stats"]["n_errored_trials"] = 1
    payload["trial_results"][0]["exception_info"] = {"exception_type": "TimeoutError"}
    result.write_text(json.dumps(payload), encoding="utf-8")

    evidence, failures = build_observation_evidence(
        dataset="mathematical-benchmarks-v1",
        condition="control",
        job_path=job_path,
        jobs_dir=tmp_path,
        result_path=result,
    )

    assert evidence["status"] == "INCOMPLETE"
    assert evidence["trials"][0]["status"] == "ERROR"
    assert any("incomplete" in f for f in failures)


def test_schema_rejects_valid_evidence_with_noncompleted_trial() -> None:
    from benchmarks.validation.observation_results_support import _evidence
    from jsonschema import Draft202012Validator

    schema = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "schemas"
            / "observation-evidence.schema.json"
        ).read_text(encoding="utf-8")
    )
    evidence = _evidence("control", [1.0])
    evidence["trials"][0]["status"] = "RUNNING"
    errors = list(Draft202012Validator(schema).iter_errors(evidence))
    assert errors, "schema must reject VALID evidence with a non-COMPLETED trial"
