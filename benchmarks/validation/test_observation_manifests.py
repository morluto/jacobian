"""Tests for manifest-driven artifact collection, source-reuse detection, and artifact path hygiene."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.tooling import observation_artifacts
from benchmarks.validation.observation_results_support import (
    _DIGEST,
    _artifact,
    _write_trial_manifest,
)

_artifact_path_failures = observation_artifacts._artifact_path_failures
_artifact_source_reuse = observation_artifacts.artifact_source_reuse
_manifest_artifacts_for_dir = observation_artifacts._manifest_artifacts_for_dir
_trial_artifacts = observation_artifacts.trial_artifacts
_trial_reasoning_protocol = observation_artifacts.trial_reasoning_protocol


def _tool_event(
    tool: str,
    arguments: dict[str, object],
    response: dict[str, object],
) -> dict[str, object]:
    return {
        "type": "item.completed",
        "item": {
            "type": "mcp_tool_call",
            "tool": tool,
            "arguments": arguments,
            "status": "completed",
            "result": {
                "isError": False,
                "content": [{"type": "text", "text": json.dumps(response)}],
            },
        },
    }


# ---------------------------------------------------------------------------
# Regression for #341: artifact identity and source reuse (manifest-driven)
# ---------------------------------------------------------------------------


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


def test_reasoning_protocol_is_extracted_without_summary_text(tmp_path: Path) -> None:
    run_id = "00000000-0000-4000-8000-000000000000"
    call_id = "11111111-1111-4111-8111-111111111111"
    events = [
        _tool_event(
            "reasoning.write",
            {"phase": "PLAN", "summary": "private plan"},
            {"run_id": run_id},
        ),
        _tool_event(
            "reasoning.write",
            {"phase": "BEFORE_TOOL", "summary": "private purpose", "run_id": run_id},
            {"run_id": run_id, "call_id": call_id},
        ),
        _tool_event(
            "math.run",
            {"reasoning_run_id": run_id, "reasoning_call_id": call_id},
            {"execution": {"status": "COMPLETED"}},
        ),
        _tool_event(
            "reasoning.write",
            {
                "phase": "AFTER_TOOL",
                "summary": "private interpretation",
                "run_id": run_id,
                "call_id": call_id,
            },
            {
                "run_id": run_id,
                "execution_status_matches": True,
                "assurance_level_matches": True,
                "completeness_status_matches": True,
            },
        ),
        _tool_event(
            "reasoning.write",
            {"phase": "FINAL", "summary": "private final", "run_id": run_id},
            {"run_id": run_id, "state": "FINALIZED"},
        ),
    ]
    trial_dir = tmp_path / "job" / "attempt-0"
    _write_trial_manifest(
        trial_dir,
        [
            {
                "source": "/logs/agent/trajectory.jsonl",
                "destination": "artifacts/logs/agent/trajectory.jsonl",
                "type": "file",
                "status": "ok",
                "service": None,
                "_content": "\n".join(json.dumps(event) for event in events) + "\n",
            }
        ],
    )
    result_path = trial_dir / "result.json"
    result_path.write_text("{}", encoding="utf-8")
    artifacts, *_ = _trial_artifacts(result_path, "attempt-0", "job.json")

    protocol = _trial_reasoning_protocol(result_path, artifacts)

    assert protocol["status"] == "COMPLETE"
    assert protocol["bound_invoke_count"] == 1
    assert protocol["summary_characters"] == len(
        "private planprivate purposeprivate interpretationprivate final"
    )
    assert "private" not in json.dumps(protocol)


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
# Regression for #343: artifact path hygiene (fail-closed path/symlink checks)
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
