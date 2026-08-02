from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from benchmarks.tooling import observation_results
from benchmarks.tooling.observation_results import (
    _artifact_path_failures,
    _artifact_source_reuse,
    _comparison_job,
    _manifest_artifacts_for_dir,
    _normalize_selection,
    _resolve_binding,
    _trial_artifacts,
    _validate_explicit_task_path,
    build_observation_evidence,
    compare_evidence,
    render_markdown,
)

_DIGEST = "sha256:" + "a" * 64
_SNAPSHOT_ID = "sha256:" + "f" * 64
_HARBOR_VERSION = "0.20.0"


def _trial(repetition: int, reward: float) -> dict:
    return {
        "task": "case",
        "task_digest": _DIGEST,
        "repetition": repetition,
        "trial_name": f"attempt-{repetition}",
        "pair_id": None,
        "status": "COMPLETED",
        "exception_type": None,
        "model": "model",
        "model_provider": None,
        "agent": {"name": "codex", "version": "1"},
        "rewards": {
            "correctness": reward,
            "evidence_validity": reward,
            "scope_accuracy": 1.0,
            "assurance_calibration": 1.0,
            "reward": reward,
            "false_certification": False,
        },
        "false_certification": False,
        "verifier_state": "COMPLETED",
        "tokens": {"input": 10, "cache": None, "output": 5},
        "cost_usd": 0.01,
        "agent_seconds": 2.0,
        "budgets": {"max_tokens": 1000, "max_cost_usd": 1.0},
        "artifacts": [],
        "tool_calls": {},
        "tool_errors": 0,
        "raw_result_digest": "sha256:" + "e" * 64,
    }


def _evidence(condition: str, correctness: list[float]) -> dict:
    return {
        "schema_version": "2",
        "evidence_class": "workflow-observation",
        "causal_claim_authorized": False,
        "status": "VALID",
        "source_sha": "a" * 40,
        "dataset": "agent-workflow-v1",
        "condition": condition,
        "snapshot_id": _SNAPSHOT_ID,
        "harbor_version": _HARBOR_VERSION,
        "eval_args": {
            "selection_mode": "dataset-task-names",
            "datasets": [
                {
                    "path": "benchmarks/datasets/agent-workflow-v1",
                    "task_names": ["case"],
                }
            ],
            "tasks": None,
            "selection": ["case"],
            "n_attempts": len(correctness),
            "selection_digest": "sha256:" + "0" * 64,
        },
        "job": {
            "path": "job.json",
            "digest": "sha256:" + "c" * 64,
            "comparison_signature": "sha256:" + "b" * 64,
            "n_attempts": len(correctness),
        },
        "runtime_snapshot": {},
        "fixed_invariants": {
            "model": "model",
            "tasks": [{"task": "case", "digest": _DIGEST}],
            "sampling_seed": None,
            "sampling_deterministic": False,
            "runtime": {},
        },
        "result": {"path": "result.json", "digest": "sha256:" + "d" * 64},
        "trials": [_trial(i, reward) for i, reward in enumerate(correctness)],
        "validation_failures": [],
    }


# ---------------------------------------------------------------------------
# Comparison behavior (preserved v2 public contract)
# ---------------------------------------------------------------------------


def test_paired_report_keeps_public_claim_boundary() -> None:
    report = compare_evidence(
        _evidence("control", [0.0, 1.0]), _evidence("treatment", [1.0, 1.0])
    )

    assert report["status"] == "VALID"
    assert report["causal_claim_authorized"] is False
    assert report["metrics"]["correctness"]["paired_delta"] == 0.5
    assert (
        report["metrics"]["correctness"]["interpretation"] == "descriptive-small-sample"
    )
    assert "does not itself authorize a causal" in render_markdown(report)


def test_comparison_rejects_invariant_drift() -> None:
    control = _evidence("control", [1.0])
    treatment = deepcopy(_evidence("treatment", [1.0]))
    treatment["fixed_invariants"]["model"] = "different"

    report = compare_evidence(control, treatment)

    assert report["status"] == "INVALID"
    assert "fixed invariants differ" in report["validation_failures"]


def test_comparison_rejects_unpaired_repetitions() -> None:
    report = compare_evidence(
        _evidence("control", [1.0, 1.0]), _evidence("treatment", [1.0])
    )

    assert report["status"] == "INVALID"
    assert (
        "control/treatment trials do not pair exactly" in report["validation_failures"]
    )


def test_comparison_rejects_duplicate_pair_keys() -> None:
    control = _evidence("control", [1.0])
    control["trials"].append(deepcopy(control["trials"][0]))

    report = compare_evidence(control, _evidence("treatment", [1.0]))

    assert report["status"] == "INVALID"
    assert "duplicate" in " ".join(report["validation_failures"])


def test_comparison_derives_heldout_class_from_both_inputs() -> None:
    control = _evidence("C1", [1.0])
    treatment = _evidence("C2", [1.0])
    control["evidence_class"] = "held-out-comparative-evaluation"
    treatment["evidence_class"] = "held-out-comparative-evaluation"

    report = compare_evidence(control, treatment)

    assert report["evidence_class"] == "held-out-comparison"
    assert report["status"] == "VALID"


def test_comparison_rejects_same_condition_inputs() -> None:
    report = compare_evidence(_evidence("control", [1.0]), _evidence("control", [1.0]))

    assert report["status"] == "INVALID"
    assert (
        "conditions must be a distinct control/treatment or C1/C2 pair"
        in report["validation_failures"]
    )


def test_comparison_normalization_allows_only_frozen_jacobian_differences() -> None:
    control = {
        "environment": {
            "extra_docker_compose": ["benchmarks/config/agent-eval-proxy.compose.yaml"]
        },
        "agents": [{"name": "codex"}],
    }
    treatment = {
        "environment": {
            "extra_docker_compose": [
                "benchmarks/config/agent-eval-proxy.compose.yaml",
                "/tmp/rendered/c2.compose.json",
            ]
        },
        "agents": [
            {
                "name": "codex",
                "mcp_servers": [
                    {
                        "name": "jacobian",
                        "transport": "streamable-http",
                        "url": "http://jacobian:8000/mcp",
                    }
                ],
            }
        ],
    }

    assert _comparison_job(control) == _comparison_job(treatment)

    treatment["environment"]["extra_docker_compose"].append("unexpected.yaml")
    assert _comparison_job(control) != _comparison_job(treatment)

    treatment["environment"]["extra_docker_compose"].pop()
    treatment["agents"][0]["mcp_servers"].append(
        {"name": "unexpected", "transport": "stdio", "url": "http://other"}
    )
    assert _comparison_job(control) != _comparison_job(treatment)


# ---------------------------------------------------------------------------
# Normalization integration (v2 bindings)
# ---------------------------------------------------------------------------


def _write_observation_job(
    tmp_path: Path,
    job: dict,
    *,
    snapshot_id: str | None = _SNAPSHOT_ID,
    harbor_version: str | None = _HARBOR_VERSION,
) -> Path:
    if snapshot_id is not None:
        job["snapshot_id"] = snapshot_id
    if harbor_version is not None:
        job["harbor_version"] = harbor_version
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(job), encoding="utf-8")
    return job_path


def _write_result(
    tmp_path: Path, *, task_name: str = "jacobian/graph-counterexample"
) -> Path:
    result = {
        "id": "job",
        "started_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T00:01:00Z",
        "n_total_trials": 1,
        "stats": {
            "n_completed_trials": 1,
            "n_errored_trials": 0,
            "n_running_trials": 0,
            "n_pending_trials": 0,
            "n_cancelled_trials": 0,
        },
        "trial_results": [
            {
                "task_name": task_name,
                "task_checksum": _DIGEST,
                "trial_name": "attempt-0",
                "agent_info": {
                    "name": "codex",
                    "version": "1",
                    "model_info": {"name": "model"},
                },
                "agent_result": {
                    "n_input_tokens": 10,
                    "n_output_tokens": 5,
                    "cost_usd": 0.01,
                },
                "verifier_result": {
                    "status": "COMPLETED",
                    "rewards": {"correctness": 1.0, "false_certification": 0.0},
                },
                "exception_info": None,
            }
        ],
    }
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    return result_path


def _write_trial_manifest(
    trial_dir: Path,
    entries: list[dict],
    *,
    artifacts_subdir: str = "artifacts",
) -> Path:
    """Write a Harbor 0.20 manifest.json and the referenced artifact files."""
    artifacts_dir = trial_dir / artifacts_subdir
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        if entry.get("status") != "ok":
            continue
        destination = entry["destination"]
        rel = destination.removeprefix("artifacts/")
        if not rel:
            continue
        host = artifacts_dir / rel
        host.parent.mkdir(parents=True, exist_ok=True)
        host.write_text(entry.get("_content", '{"events": []}'), encoding="utf-8")
    manifest_path = artifacts_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {k: v for k, v in entry.items() if not k.startswith("_")}
                for entry in entries
            ]
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_observation_normalization_binds_v2_fields(
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
    )

    assert failures == []
    assert evidence["status"] == "VALID"
    assert evidence["schema_version"] == "2"
    assert evidence["fixed_invariants"]["model"] == "model"
    assert evidence["eval_args"]["selection_mode"] == "dataset-task-names"
    assert evidence["eval_args"]["selection"] == ["graph-counterexample"]
    assert evidence["snapshot_id"] == _SNAPSHOT_ID
    assert evidence["harbor_version"] == _HARBOR_VERSION
    trial = evidence["trials"][0]
    assert trial["agent"] == {"name": "codex", "version": "1"}
    assert trial["verifier_state"] == "COMPLETED"
    assert trial["budgets"] is None
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


# ---------------------------------------------------------------------------
# Regression for #338: strict dataset/task selection normalization
# ---------------------------------------------------------------------------


def _selection_fixture(tmp_path: Path) -> tuple[dict, dict, Path]:
    known = {"case": _DIGEST, "other": "sha256:" + "b" * 64}
    task_dirs: dict[str, Path] = {}
    for name in known:
        task_dir = tmp_path / name
        task_dir.mkdir()
        (task_dir / "task.toml").write_text("version = '1'\n", encoding="utf-8")
        task_dirs[name] = task_dir
    return known, task_dirs, tmp_path


def test_selection_rejects_mixed_datasets_and_tasks(tmp_path: Path) -> None:
    known, task_dirs, dataset_path = _selection_fixture(tmp_path)
    job = {
        "datasets": [{"path": "x", "task_names": ["case"]}],
        "tasks": [{"path": "case"}],
    }

    selected, mode, _eval, failures = _normalize_selection(
        job, known=known, task_dirs=task_dirs, dataset_path=dataset_path
    )

    assert mode == "mixed"
    assert selected == []
    assert any("not both" in f for f in failures)


def test_selection_rejects_implicit_fallback(tmp_path: Path) -> None:
    known, task_dirs, dataset_path = _selection_fixture(tmp_path)
    job = {"n_attempts": 1}

    selected, mode, _eval, failures = _normalize_selection(
        job, known=known, task_dirs=task_dirs, dataset_path=dataset_path
    )

    assert mode == "implicit-fallback"
    assert selected == []
    assert any("implicit fallback" in f for f in failures)


def test_selection_rejects_unknown_task_name(tmp_path: Path) -> None:
    known, task_dirs, dataset_path = _selection_fixture(tmp_path)
    job = {"datasets": [{"path": "x", "task_names": ["nonexistent"]}]}

    _selected, _mode, _eval, failures = _normalize_selection(
        job, known=known, task_dirs=task_dirs, dataset_path=dataset_path
    )

    assert any("unknown task name" in f for f in failures)


def test_selection_rejects_empty_datasets_and_empty_task_names(tmp_path: Path) -> None:
    known, task_dirs, dataset_path = _selection_fixture(tmp_path)

    _s, _m, _e, empty_failures = _normalize_selection(
        {"datasets": []}, known=known, task_dirs=task_dirs, dataset_path=dataset_path
    )
    assert any("non-empty array" in f for f in empty_failures)

    _s, _m, _e, names_failures = _normalize_selection(
        {"datasets": [{"path": "x", "task_names": []}]},
        known=known,
        task_dirs=task_dirs,
        dataset_path=dataset_path,
    )
    assert any("non-empty array" in f for f in names_failures)


def test_selection_optional_task_names_selects_all_known(tmp_path: Path) -> None:
    known, task_dirs, dataset_path = _selection_fixture(tmp_path)
    job = {"datasets": [{"path": "x"}]}

    selected, mode, _eval, failures = _normalize_selection(
        job, known=known, task_dirs=task_dirs, dataset_path=dataset_path
    )

    assert failures == []
    assert mode == "dataset-task-names"
    assert selected == ["case", "other"]


def test_selection_explicit_tasks_reject_outside_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    known, task_dirs, _ = _selection_fixture(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "task.toml").write_text("version = '1'\n", encoding="utf-8")
    monkeypatch.setattr(observation_results, "ROOT", tmp_path)
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()

    _s, _m, _e, failures = _normalize_selection(
        {"tasks": [{"path": "outside"}]},
        known=known,
        task_dirs=task_dirs,
        dataset_path=dataset_path,
    )

    assert any("outside the dataset" in f for f in failures)


def test_selection_explicit_tasks_reject_unknown_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    known, task_dirs, _ = _selection_fixture(tmp_path)
    monkeypatch.setattr(observation_results, "ROOT", tmp_path)
    bogus = tmp_path / "bogus"
    bogus.mkdir()
    (bogus / "task.toml").write_text("version = '1'\n", encoding="utf-8")

    _s, _m, _e, failures = _normalize_selection(
        {"tasks": [{"path": "bogus"}]},
        known=known,
        task_dirs=task_dirs,
        dataset_path=tmp_path,
    )

    assert any("not a known task" in f for f in failures)


def test_selection_explicit_tasks_reject_absolute_and_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    known, task_dirs, _ = _selection_fixture(tmp_path)
    monkeypatch.setattr(observation_results, "ROOT", tmp_path)

    _s, _m, _e, abs_failures = _normalize_selection(
        {"tasks": [{"path": "/etc/passwd"}]},
        known=known,
        task_dirs=task_dirs,
        dataset_path=tmp_path,
    )
    assert any("must be relative" in f for f in abs_failures)

    _s, _m, _e, trav_failures = _normalize_selection(
        {"tasks": [{"path": "../case"}]},
        known=known,
        task_dirs=task_dirs,
        dataset_path=tmp_path,
    )
    assert any("traverse" in f for f in trav_failures)


def test_selection_explicit_tasks_reject_missing_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    known, _task_dirs, _ = _selection_fixture(tmp_path)
    monkeypatch.setattr(observation_results, "ROOT", tmp_path)
    no_manifest = tmp_path / "case"
    # Reuse the "case" dir from the fixture but remove its manifest.
    (no_manifest / "task.toml").unlink()
    task_dirs = {"case": no_manifest}

    _s, _m, _e, failures = _normalize_selection(
        {"tasks": [{"path": "case"}]},
        known=known,
        task_dirs=task_dirs,
        dataset_path=tmp_path,
    )

    assert any("missing its manifest" in f for f in failures)


def test_selection_explicit_tasks_reject_escaping_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    known, task_dirs, _ = _selection_fixture(tmp_path)
    monkeypatch.setattr(observation_results, "ROOT", tmp_path)
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "case")

    _s, _m, _e, failures = _normalize_selection(
        {"tasks": [{"path": "link"}]},
        known=known,
        task_dirs=task_dirs,
        dataset_path=tmp_path,
    )

    assert any("symlink" in f for f in failures)


def test_selection_explicit_tasks_accept_valid_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    known, task_dirs, _ = _selection_fixture(tmp_path)
    monkeypatch.setattr(observation_results, "ROOT", tmp_path)

    selected, mode, _eval, failures = _normalize_selection(
        {"tasks": [{"path": "case"}]},
        known=known,
        task_dirs=task_dirs,
        dataset_path=tmp_path,
    )

    assert failures == []
    assert mode == "explicit-tasks"
    assert selected == ["case"]


# ---------------------------------------------------------------------------
# Regression for #341: artifact identity and source reuse (manifest-driven)
# ---------------------------------------------------------------------------


def _artifact(source_path: str, trial: str, digest: str) -> dict:
    return {
        "job": "job.json",
        "trial": trial,
        "step": 0,
        "step_name": None,
        "source_path": source_path,
        "manifest_source": "/app/trace.json",
        "service": None,
        "artifact_path": source_path,
        "digest": digest,
    }


def test_artifact_source_reuse_is_rejected() -> None:
    # Same canonical host source_path across two trials is a reuse violation.
    trials = [
        {
            "trial_name": "a",
            "artifacts": [_artifact("artifacts/trace.json", "a", _DIGEST)],
        },
        {
            "trial_name": "b",
            "artifacts": [_artifact("artifacts/trace.json", "b", _DIGEST)],
        },
    ]

    failures = _artifact_source_reuse(trials)

    assert any("source path reused" in f for f in failures)


def test_identical_bytes_at_distinct_source_paths_allowed() -> None:
    # Independent trials have distinct host source_paths even when the Harbor
    # manifest_source (container path) is identical.
    trials = [
        {
            "trial_name": "a",
            "artifacts": [
                _artifact("trial-a/artifacts/trace.json", "a", _DIGEST),
            ],
        },
        {
            "trial_name": "b",
            "artifacts": [
                _artifact("trial-b/artifacts/trace.json", "b", _DIGEST),
            ],
        },
    ]

    failures = _artifact_source_reuse(trials)

    assert failures == []


def test_same_manifest_source_distinct_host_path_allowed() -> None:
    # The Harbor manifest source is a container path that repeats across
    # independent trial containers; reuse detection must use source_path, not
    # manifest_source.
    trials = [
        {
            "trial_name": "a",
            "artifacts": [
                {
                    "job": "job.json",
                    "trial": "a",
                    "step": 0,
                    "step_name": None,
                    "source_path": "trial-a/artifacts/logs/agent/trajectory.json",
                    "manifest_source": "/logs/agent/trajectory.json",
                    "service": None,
                    "artifact_path": "logs/agent/trajectory.json",
                    "digest": _DIGEST,
                }
            ],
        },
        {
            "trial_name": "b",
            "artifacts": [
                {
                    "job": "job.json",
                    "trial": "b",
                    "step": 0,
                    "step_name": None,
                    "source_path": "trial-b/artifacts/logs/agent/trajectory.json",
                    "manifest_source": "/logs/agent/trajectory.json",
                    "service": None,
                    "artifact_path": "logs/agent/trajectory.json",
                    "digest": _DIGEST,
                }
            ],
        },
    ]

    failures = _artifact_source_reuse(trials)

    assert failures == []


def test_equal_manifest_source_across_repetitions_accepted(
    tmp_path: Path,
) -> None:
    # Two independent trial containers under one job dir collect the same
    # Harbor manifest source (/logs/agent/trajectory.json).  Their canonical
    # host source_paths include the trial directory name, so they differ and
    # reuse detection accepts them as independent.
    job_dir = tmp_path / "job-output"
    job_dir.mkdir()
    for trial_name in ("attempt-0", "attempt-1"):
        trial_dir = job_dir / trial_name
        _write_trial_manifest(
            trial_dir,
            [
                {
                    "source": "/logs/agent/trajectory.json",
                    "destination": "artifacts/logs/agent/trajectory.json",
                    "type": "file",
                    "status": "ok",
                    "service": None,
                    "_content": '{"events": []}',
                }
            ],
        )
        (trial_dir / "result.json").write_text("{}", encoding="utf-8")

    artifacts_0, _c0, _e0, failures_0 = _trial_artifacts(
        job_dir / "attempt-0" / "result.json", "attempt-0", "job.json"
    )
    artifacts_1, _c1, _e1, failures_1 = _trial_artifacts(
        job_dir / "attempt-1" / "result.json", "attempt-1", "job.json"
    )

    assert failures_0 == []
    assert failures_1 == []
    assert len(artifacts_0) == 1
    assert len(artifacts_1) == 1
    # Same Harbor manifest_source (container path) ...
    assert artifacts_0[0]["manifest_source"] == "/logs/agent/trajectory.json"
    assert artifacts_1[0]["manifest_source"] == "/logs/agent/trajectory.json"
    # ... but distinct job-directory-relative source_paths including trial name.
    assert artifacts_0[0]["source_path"] == (
        "attempt-0/artifacts/logs/agent/trajectory.json"
    )
    assert artifacts_1[0]["source_path"] == (
        "attempt-1/artifacts/logs/agent/trajectory.json"
    )
    # Reuse detection on source_path accepts both.
    trials = [
        {"trial_name": "attempt-0", "artifacts": artifacts_0},
        {"trial_name": "attempt-1", "artifacts": artifacts_1},
    ]
    assert _artifact_source_reuse(trials) == []


def test_heldout_artifact_source_paths_include_pair_and_condition(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / "job-output"
    job_dir.mkdir()
    for trial_name in ("attempt-0", "attempt-1"):
        trial_dir = job_dir / trial_name
        _write_trial_manifest(
            trial_dir,
            [
                {
                    "source": "/logs/agent/trajectory.json",
                    "destination": "artifacts/logs/agent/trajectory.json",
                    "type": "file",
                    "status": "ok",
                    "service": None,
                    "_content": '{"events": []}',
                }
            ],
        )
        (trial_dir / "result.json").write_text("{}", encoding="utf-8")

    artifacts_0, *_ = _trial_artifacts(
        job_dir / "attempt-0" / "result.json",
        "attempt-0",
        "job.json",
        source_prefix="copy-token-0-r001/C1",
    )
    artifacts_1, *_ = _trial_artifacts(
        job_dir / "attempt-1" / "result.json",
        "attempt-1",
        "job.json",
        source_prefix="copy-token-0-r002/C1",
    )

    assert artifacts_0[0]["source_path"].startswith("copy-token-0-r001/C1/")
    assert artifacts_1[0]["source_path"].startswith("copy-token-0-r002/C1/")
    assert (
        _artifact_source_reuse(
            [
                {"trial_name": "attempt-0", "artifacts": artifacts_0},
                {"trial_name": "attempt-1", "artifacts": artifacts_1},
            ]
        )
        == []
    )


# ---------------------------------------------------------------------------
# Manifest-driven artifact collection (single-step and multi-step)
# ---------------------------------------------------------------------------


def test_single_step_manifest_binds_artifact_identity(tmp_path: Path) -> None:
    trial_dir = tmp_path / "trial-0"
    trial_dir.mkdir()
    _write_trial_manifest(
        trial_dir,
        [
            {
                "source": "/logs/agent/trajectory.json",
                "destination": "artifacts/logs/agent/trajectory.json",
                "type": "file",
                "status": "ok",
                "service": None,
                "_content": '{"events": []}',
            }
        ],
    )
    result_path = trial_dir / "result.json"
    result_path.write_text("{}", encoding="utf-8")

    artifacts, _calls, _errors, failures = _trial_artifacts(
        result_path, "attempt-0", "job.json"
    )

    assert failures == []
    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact["job"] == "job.json"
    assert artifact["trial"] == "attempt-0"
    assert artifact["step"] == 0
    assert artifact["step_name"] is None
    assert artifact["source_path"] == "trial-0/artifacts/logs/agent/trajectory.json"
    assert artifact["manifest_source"] == "/logs/agent/trajectory.json"
    assert artifact["service"] is None
    assert artifact["artifact_path"] == "logs/agent/trajectory.json"
    assert artifact["digest"].startswith("sha256:")


def test_single_step_manifest_preserves_sidecar_service(tmp_path: Path) -> None:
    trial_dir = tmp_path / "trial-0"
    trial_dir.mkdir()
    _write_trial_manifest(
        trial_dir,
        [
            {
                "source": "/var/log/api/requests.log",
                "destination": "artifacts/var/log/api/requests.log",
                "type": "file",
                "status": "ok",
                "service": "api",
                "_content": "GET /",
            }
        ],
    )
    result_path = trial_dir / "result.json"
    result_path.write_text("{}", encoding="utf-8")

    artifacts, _calls, _errors, failures = _trial_artifacts(
        result_path, "attempt-0", "job.json"
    )

    assert failures == []
    assert len(artifacts) == 1
    assert artifacts[0]["service"] == "api"
    assert artifacts[0]["manifest_source"] == "/var/log/api/requests.log"
    assert artifacts[0]["source_path"] == "trial-0/artifacts/var/log/api/requests.log"


def test_multi_step_manifest_binds_per_step_identity(tmp_path: Path) -> None:
    trial_dir = tmp_path / "trial-0"
    trial_dir.mkdir()
    steps = trial_dir / "steps"
    for step_name in ("scaffold", "document"):
        step_dir = steps / step_name
        _write_trial_manifest(
            step_dir,
            [
                {
                    "source": f"/app/{step_name}.json",
                    "destination": f"artifacts/app/{step_name}.json",
                    "type": "file",
                    "status": "ok",
                    "service": None,
                    "_content": f'{{"step": "{step_name}"}}',
                }
            ],
        )
    result_path = trial_dir / "result.json"
    result_path.write_text("{}", encoding="utf-8")

    artifacts, _calls, _errors, failures = _trial_artifacts(
        result_path, "attempt-0", "job.json"
    )

    assert failures == []
    assert len(artifacts) == 2
    assert artifacts[0]["step"] == 0
    assert artifacts[0]["step_name"] == "document"
    assert artifacts[0]["manifest_source"] == "/app/document.json"
    assert artifacts[0]["source_path"] == (
        "trial-0/steps/document/artifacts/app/document.json"
    )
    assert artifacts[1]["step"] == 1
    assert artifacts[1]["step_name"] == "scaffold"
    assert artifacts[1]["manifest_source"] == "/app/scaffold.json"
    assert artifacts[1]["source_path"] == (
        "trial-0/steps/scaffold/artifacts/app/scaffold.json"
    )


def test_directory_manifest_enumerates_regular_files(tmp_path: Path) -> None:
    trial_dir = tmp_path / "trial-0"
    trial_dir.mkdir()
    artifacts_dir = trial_dir / "artifacts"
    convention = artifacts_dir / "logs" / "artifacts"
    convention.mkdir(parents=True)
    (convention / "trajectory.json").write_text('{"events": []}', encoding="utf-8")
    (convention / "telemetry.json").write_text('{"metrics": {}}', encoding="utf-8")
    manifest = [
        {
            "source": "/logs/artifacts",
            "destination": "artifacts/logs/artifacts",
            "type": "directory",
            "status": "ok",
            "service": None,
        }
    ]
    (artifacts_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result_path = trial_dir / "result.json"
    result_path.write_text("{}", encoding="utf-8")

    artifacts, _calls, _errors, failures = _trial_artifacts(
        result_path, "attempt-0", "job.json"
    )

    assert failures == []
    assert len(artifacts) == 2
    paths = {a["artifact_path"] for a in artifacts}
    assert paths == {"logs/artifacts/telemetry.json", "logs/artifacts/trajectory.json"}
    for artifact in artifacts:
        assert artifact["manifest_source"] == "/logs/artifacts"
        assert artifact["source_path"].startswith("trial-0/artifacts/logs/artifacts/")
        assert artifact["service"] is None
        assert artifact["digest"].startswith("sha256:")


def test_directory_manifest_empty_ok_is_failure(tmp_path: Path) -> None:
    trial_dir = tmp_path / "trial-0"
    trial_dir.mkdir()
    artifacts_dir = trial_dir / "artifacts"
    empty_dir = artifacts_dir / "logs" / "artifacts"
    empty_dir.mkdir(parents=True)
    manifest = [
        {
            "source": "/logs/artifacts",
            "destination": "artifacts/logs/artifacts",
            "type": "directory",
            "status": "ok",
            "service": None,
        }
    ]
    (artifacts_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result_path = trial_dir / "result.json"
    result_path.write_text("{}", encoding="utf-8")

    artifacts, _calls, _errors, failures = _trial_artifacts(
        result_path, "attempt-0", "job.json"
    )

    assert artifacts == []
    assert any("unexpectedly empty" in f for f in failures)


def test_directory_manifest_missing_dir_is_failure(tmp_path: Path) -> None:
    trial_dir = tmp_path / "trial-0"
    trial_dir.mkdir()
    artifacts_dir = trial_dir / "artifacts"
    artifacts_dir.mkdir()
    manifest = [
        {
            "source": "/logs/artifacts",
            "destination": "artifacts/logs/artifacts",
            "type": "directory",
            "status": "ok",
            "service": None,
        }
    ]
    (artifacts_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result_path = trial_dir / "result.json"
    result_path.write_text("{}", encoding="utf-8")

    artifacts, _calls, _errors, failures = _trial_artifacts(
        result_path, "attempt-0", "job.json"
    )

    assert artifacts == []
    assert any("directory is missing on disk" in f for f in failures)


def test_manifest_missing_manifest_is_failure(tmp_path: Path) -> None:
    trial_dir = tmp_path / "trial-0"
    trial_dir.mkdir()
    (trial_dir / "artifacts").mkdir()
    result_path = trial_dir / "result.json"
    result_path.write_text("{}", encoding="utf-8")

    _artifacts, _calls, _errors, failures = _trial_artifacts(
        result_path, "attempt-0", "job.json"
    )

    assert any("manifest is missing" in f for f in failures)


def test_manifest_missing_artifacts_dir_is_failure(tmp_path: Path) -> None:
    trial_dir = tmp_path / "trial-0"
    trial_dir.mkdir()
    result_path = trial_dir / "result.json"
    result_path.write_text("{}", encoding="utf-8")

    _artifacts, _calls, _errors, failures = _trial_artifacts(
        result_path, "attempt-0", "job.json"
    )

    assert any("artifacts directory is missing" in f for f in failures)


def test_manifest_missing_file_on_disk_is_failure(tmp_path: Path) -> None:
    trial_dir = tmp_path / "trial-0"
    trial_dir.mkdir()
    artifacts_dir = trial_dir / "artifacts"
    artifacts_dir.mkdir()
    manifest = [
        {
            "source": "/app/trace.json",
            "destination": "artifacts/app/trace.json",
            "type": "file",
            "status": "ok",
            "service": None,
        }
    ]
    (artifacts_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result_path = trial_dir / "result.json"
    result_path.write_text("{}", encoding="utf-8")

    _artifacts, _calls, _errors, failures = _trial_artifacts(
        result_path, "attempt-0", "job.json"
    )

    assert any("file is missing on disk" in f for f in failures)


def test_manifest_non_conclusion_status_is_failure(tmp_path: Path) -> None:
    trial_dir = tmp_path / "trial-0"
    trial_dir.mkdir()
    _write_trial_manifest(
        trial_dir,
        [
            {
                "source": "/app/trace.json",
                "destination": "artifacts/app/trace.json",
                "type": "file",
                "status": "failed",
                "service": None,
            }
        ],
    )
    result_path = trial_dir / "result.json"
    result_path.write_text("{}", encoding="utf-8")

    artifacts, _calls, _errors, failures = _trial_artifacts(
        result_path, "attempt-0", "job.json"
    )

    assert artifacts == []
    assert any("non-conclusion status" in f for f in failures)


def test_manifest_malformed_entry_is_failure(tmp_path: Path) -> None:
    trial_dir = tmp_path / "trial-0"
    trial_dir.mkdir()
    artifacts_dir = trial_dir / "artifacts"
    artifacts_dir.mkdir()
    (artifacts_dir / "manifest.json").write_text(
        json.dumps(
            [{"source": "", "destination": "x", "type": "file", "status": "ok"}]
        ),
        encoding="utf-8",
    )
    result_path = trial_dir / "result.json"
    result_path.write_text("{}", encoding="utf-8")

    _artifacts, _calls, _errors, failures = _trial_artifacts(
        result_path, "attempt-0", "job.json"
    )

    assert any("source must be a non-empty string" in f for f in failures)


def test_manifest_not_array_is_failure(tmp_path: Path) -> None:
    trial_dir = tmp_path / "trial-0"
    trial_dir.mkdir()
    artifacts_dir = trial_dir / "artifacts"
    artifacts_dir.mkdir()
    (artifacts_dir / "manifest.json").write_text("{}", encoding="utf-8")
    result_path = trial_dir / "result.json"
    result_path.write_text("{}", encoding="utf-8")

    _artifacts, _calls, _errors, failures = _trial_artifacts(
        result_path, "attempt-0", "job.json"
    )

    assert any("must be a JSON array" in f for f in failures)


def test_manifest_empty_array_is_failure(tmp_path: Path) -> None:
    trial_dir = tmp_path / "trial-0"
    trial_dir.mkdir()
    artifacts_dir = trial_dir / "artifacts"
    artifacts_dir.mkdir()
    (artifacts_dir / "manifest.json").write_text("[]", encoding="utf-8")
    result_path = trial_dir / "result.json"
    result_path.write_text("{}", encoding="utf-8")

    _artifacts, _calls, _errors, failures = _trial_artifacts(
        result_path, "attempt-0", "job.json"
    )

    assert any("manifest is empty" in f for f in failures)


def test_manifest_artifacts_for_dir_rejects_symlink_manifest(
    tmp_path: Path,
) -> None:
    trial_dir = tmp_path / "trial-0"
    trial_dir.mkdir()
    artifacts_dir = trial_dir / "artifacts"
    artifacts_dir.mkdir()
    outside = tmp_path / "outside-manifest.json"
    outside.write_text("[]", encoding="utf-8")
    (artifacts_dir / "manifest.json").symlink_to(outside)

    _artifacts, failures = _manifest_artifacts_for_dir(
        artifacts_dir,
        trial_root=trial_dir,
        trial_name="attempt-0",
        job_label="job.json",
        step_index=0,
        step_name=None,
    )

    assert any("forbidden symlink" in f for f in failures)


# ---------------------------------------------------------------------------
# Regression for #343: artifact path hygiene
# ---------------------------------------------------------------------------


def test_artifact_rejects_escaping_symlink(tmp_path: Path) -> None:
    trial_root = tmp_path / "trial"
    trial_root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("secret", encoding="utf-8")
    link = trial_root / "trajectory.json"
    link.symlink_to(outside)

    failures = _artifact_path_failures(link, trial_root)

    assert any("symlink" in f for f in failures)


def test_artifact_rejects_escape_outside_trial_root(tmp_path: Path) -> None:
    trial_root = tmp_path / "trial"
    trial_root.mkdir()
    outside = tmp_path / "trajectory.json"
    outside.write_text("x", encoding="utf-8")

    failures = _artifact_path_failures(outside, trial_root)

    assert any("escapes trial root" in f for f in failures)


def test_artifact_accepts_clean_relative_path(tmp_path: Path) -> None:
    trial_root = tmp_path / "trial"
    nested = trial_root / "steps" / "step-0"
    nested.mkdir(parents=True)
    clean = nested / "trajectory.json"
    clean.write_text("{}", encoding="utf-8")

    failures = _artifact_path_failures(clean, trial_root)

    assert failures == []


def test_explicit_task_path_rejects_absolute() -> None:
    _short, failures = _validate_explicit_task_path(
        "/etc/passwd", dataset_path=None, task_dirs={}
    )

    assert _short is None
    assert any("must be relative" in f for f in failures)


def test_explicit_task_path_rejects_traversal() -> None:
    _short, failures = _validate_explicit_task_path(
        "../escape", dataset_path=None, task_dirs={}
    )

    assert _short is None
    assert any("traverse" in f for f in failures)


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
