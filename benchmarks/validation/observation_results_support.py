"""Shared constants and fixture helpers for observation-results validation tests."""

from __future__ import annotations

import json
from pathlib import Path

_DIGEST = "sha256:" + "a" * 64
_SNAPSHOT_ID = "sha256:" + "f" * 64
_HARBOR_VERSION = "0.20.0"
_JACOBIAN_IMAGE = {
    "source_sha": "b" * 40,
    "source_dirty": False,
    "reference": "ghcr.io/morluto/jacobian:sha-" + "b" * 40,
    "digest_reference": "ghcr.io/morluto/jacobian@sha256:" + "1" * 64,
    "image_id": "sha256:" + "2" * 64,
    "platform": "linux/amd64",
    "jacobian_package_version": "0.8.0",
}


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
        "reasoning_protocol": {
            "status": "INCOMPLETE",
            "mode": "UNKNOWN",
            "requirement_status": "NOT_REQUIRED",
            "plan_count": 0,
            "before_tool_count": 0,
            "after_tool_count": 0,
            "final_count": 0,
            "run_count": 0,
            "bound_invoke_count": 0,
            "missing_after_tool_count": 0,
            "pending_call_count": 0,
            "unavailable_after_tool_count": 0,
            "reported_actual_mismatch_count": 0,
            "summary_characters": 0,
        },
        "raw_result_digest": "sha256:" + "e" * 64,
    }


def _evidence(condition: str, correctness: list[float]) -> dict:
    return {
        "schema_version": "3",
        "evidence_class": "workflow-observation",
        "causal_claim_authorized": False,
        "status": "VALID",
        "source_sha": "a" * 40,
        "dataset": "mathematical-benchmarks-v1",
        "condition": condition,
        "snapshot_id": _SNAPSHOT_ID,
        "harbor_version": _HARBOR_VERSION,
        "eval_args": {
            "selection_mode": "dataset-task-names",
            "datasets": [
                {
                    "path": "benchmarks/datasets/mathematical-benchmarks-v1",
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
