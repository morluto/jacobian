"""Tests for observation evidence field binding and fail-closed behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.tooling import observation_results
from benchmarks.tooling.observation_results import (
    _resolve_binding,
    build_observation_evidence,
)
from benchmarks.validation.observation_results_support import (
    _HARBOR_VERSION,
    _SNAPSHOT_ID,
    _write_observation_job,
    _write_result,
)

# ---------------------------------------------------------------------------
# Normalization integration
# ---------------------------------------------------------------------------


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
                "path": "benchmarks/datasets/agent-workflow-v1",
                "task_names": ["graph-counterexample"],
            }
        ],
    }
    job_path = _write_observation_job(tmp_path, job)
    result_path = _write_result(tmp_path)

    evidence, failures = build_observation_evidence(
        dataset="agent-workflow-v1",
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
    assert evidence["schema_version"] == "3"
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
                "path": "benchmarks/datasets/agent-workflow-v1",
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
        "condition": {
            "id": "treatment",
            "role": "PRIMARY_TREATMENT",
            "jacobian_enabled": True,
            "reasoning_log_mode": "OFF",
        },
    }

    evidence, failures = build_observation_evidence(
        dataset="agent-workflow-v1",
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


def test_required_reasoning_protocol_fails_closed_when_trace_is_missing(
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
                "path": "benchmarks/datasets/agent-workflow-v1",
                "task_names": ["graph-counterexample"],
            }
        ],
    }
    runtime = {
        "snapshot_id": _SNAPSHOT_ID,
        "harbor_version": _HARBOR_VERSION,
        "model": "model",
        "condition": {
            "id": "treatment",
            "role": "PRIMARY_TREATMENT",
            "jacobian_enabled": True,
            "reasoning_log_mode": "REQUIRED",
        },
    }
    evidence, failures = build_observation_evidence(
        dataset="agent-workflow-v1",
        condition="treatment",
        job_path=_write_observation_job(tmp_path, job),
        jobs_dir=tmp_path,
        result_path=_write_result(tmp_path),
        runtime_snapshot=runtime,
    )

    assert evidence["status"] == "INCOMPLETE"
    assert evidence["trials"][0]["reasoning_protocol"]["mode"] == "REQUIRED"
    assert "required reasoning protocol is incomplete" in " ".join(failures)


def test_unbound_reasoning_mode_with_jacobian_enabled_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Jacobian-enabled run with no reasoning_log_mode binding must not be VALID."""

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
                "path": "benchmarks/datasets/agent-workflow-v1",
                "task_names": ["graph-counterexample"],
            }
        ],
    }
    runtime = {
        "snapshot_id": _SNAPSHOT_ID,
        "harbor_version": _HARBOR_VERSION,
        "model": "model",
        "condition": {
            "id": "treatment",
            "role": "PRIMARY_TREATMENT",
            "jacobian_enabled": True,
        },
    }
    evidence, failures = build_observation_evidence(
        dataset="agent-workflow-v1",
        condition="treatment",
        job_path=_write_observation_job(tmp_path, job),
        jobs_dir=tmp_path,
        result_path=_write_result(tmp_path),
        runtime_snapshot=runtime,
    )

    assert evidence["status"] == "INCOMPLETE"
    assert evidence["trials"][0]["reasoning_protocol"]["mode"] == "UNKNOWN"
    assert (
        evidence["trials"][0]["reasoning_protocol"]["requirement_status"]
        == "INCOMPLETE"
    )
    assert "reasoning_log_mode is unbound or invalid" in " ".join(failures)


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
                "path": "benchmarks/datasets/agent-workflow-v1",
                "task_names": ["graph-counterexample"],
            }
        ],
    }
    job_path = _write_observation_job(
        tmp_path, job, snapshot_id=None, harbor_version=_HARBOR_VERSION
    )
    result_path = _write_result(tmp_path)

    evidence, failures = build_observation_evidence(
        dataset="agent-workflow-v1",
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
                "path": "benchmarks/datasets/agent-workflow-v1",
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
        dataset="agent-workflow-v1",
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
                "path": "benchmarks/datasets/agent-workflow-v1",
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
        dataset="agent-workflow-v1",
        condition="control",
        job_path=job_path,
        jobs_dir=tmp_path,
        result_path=result,
    )

    assert evidence["status"] == "INCOMPLETE"
    assert evidence["trials"][0]["status"] == "ERROR"
    assert any("incomplete" in f for f in failures)
