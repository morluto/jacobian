from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from benchmarks.tooling import observation_results
from benchmarks.tooling.observation_results import (
    build_observation_evidence,
    compare_evidence,
    render_markdown,
)


def _evidence(condition: str, correctness: list[float]) -> dict:
    trials = []
    for repetition, reward in enumerate(correctness):
        trials.append(
            {
                "task": "case",
                "task_digest": "sha256:" + "a" * 64,
                "repetition": repetition,
                "rewards": {
                    "correctness": reward,
                    "evidence_validity": reward,
                    "scope_accuracy": 1.0,
                    "assurance_calibration": 1.0,
                    "reward": reward,
                    "false_certification": False,
                },
                "false_certification": 0.0,
                "tokens": {"input": 10, "output": 5},
                "cost_usd": 0.01,
                "agent_seconds": 2.0,
            }
        )
    return {
        "schema_version": "1",
        "evidence_class": "workflow-observation",
        "status": "VALID",
        "causal_claim_authorized": False,
        "source_sha": "a" * 40,
        "dataset": "agent-workflow-v1",
        "condition": condition,
        "job": {
            "path": "job.json",
            "digest": "sha256:" + "c" * 64,
            "comparison_signature": "sha256:" + "b" * 64,
            "n_attempts": len(correctness),
        },
        "runtime_snapshot": {},
        "fixed_invariants": {
            "model": "model",
            "tasks": [{"task": "case", "digest": "sha256:" + "a" * 64}],
            "sampling_seed": None,
            "sampling_deterministic": False,
        },
        "result": {
            "path": "result.json",
            "digest": "sha256:" + ("d" if condition in {"control", "C1"} else "e") * 64,
        },
        "trials": trials,
        "validation_failures": [],
    }


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


def test_comparison_rejects_reused_result_artifact() -> None:
    control = _evidence("control", [1.0])
    treatment = _evidence("treatment", [1.0])
    shared_digest = "sha256:" + "f" * 64
    control["trials"][0]["raw_result_digest"] = shared_digest
    treatment["trials"][0]["raw_result_digest"] = shared_digest

    report = compare_evidence(control, treatment)

    assert report["status"] == "INVALID"
    assert any("normalized trial artifact" in item for item in report["validation_failures"])


def test_comparison_derives_heldout_class_from_both_inputs() -> None:
    control = _evidence("C1", [1.0])
    treatment = _evidence("C2", [1.0])
    control["evidence_class"] = "held-out-comparative-evaluation"
    treatment["evidence_class"] = "held-out-comparative-evaluation"

    report = compare_evidence(control, treatment)

    assert report["evidence_class"] == "held-out-comparison"
    assert report["status"] == "VALID"


def test_comparison_preserves_unrelated_compose_differences() -> None:
    from benchmarks.tooling.observation_results import _comparison_job

    control_job = {
        "jobs_dir": "results",
        "environment": {
            "extra_docker_compose": [
                "/abs/path/agent-eval-proxy.compose.yaml",
            ]
        },
        "agents": [{"name": "codex"}],
    }
    treatment_job = {
        "jobs_dir": "results",
        "environment": {
            "extra_docker_compose": [
                "/abs/path/agent-eval-proxy.compose.yaml",
                "/abs/path/c2.compose.json",
            ]
        },
        "agents": [{"name": "codex", "mcp_servers": [{"name": "jacobian"}]}],
    }
    # Stripping only the c2 treatment overlay and mcp_servers makes the frozen
    # control and treatment jobs compare equal.
    assert _comparison_job(deepcopy(control_job)) == _comparison_job(
        deepcopy(treatment_job)
    )

    # An unrelated sidecar survives normalization, so the signatures differ and
    # the comparison rejects the configuration drift instead of hiding it.
    with_sidecar = deepcopy(control_job)
    with_sidecar["environment"]["extra_docker_compose"].append(
        "/abs/path/extra.compose.json"
    )
    assert _comparison_job(with_sidecar) != _comparison_job(control_job)

    report = compare_evidence(
        _evidence("control", [1.0]), _evidence("treatment", [1.0])
    )
    assert report["status"] == "VALID"


def test_comparison_rejects_missing_required_outcome_dimensions() -> None:
    control = _evidence("control", [1.0])
    treatment = _evidence("treatment", [1.0])
    for trial in control["trials"] + treatment["trials"]:
        del trial["rewards"]["evidence_validity"]
        del trial["rewards"]["scope_accuracy"]
        del trial["rewards"]["assurance_calibration"]

    report = compare_evidence(control, treatment)

    assert report["status"] == "INVALID"
    failures = " ".join(report["validation_failures"])
    assert "evidence_validity" in failures
    assert "scope_accuracy" in failures
    assert "assurance_calibration" in failures


def test_comparison_binds_agent_and_provider_identity() -> None:
    control = _evidence("control", [1.0])
    treatment = deepcopy(_evidence("treatment", [1.0]))
    control["fixed_invariants"]["agent_name"] = "codex"
    control["fixed_invariants"]["agent_version"] = "1.2.3"
    control["fixed_invariants"]["model_provider"] = "openai"
    treatment["fixed_invariants"]["agent_name"] = "codex"
    treatment["fixed_invariants"]["agent_version"] = "1.2.3"
    treatment["fixed_invariants"]["model_provider"] = "openai"

    report = compare_evidence(control, treatment)

    assert report["status"] == "VALID"


def test_comparison_rejects_drifted_agent_identity() -> None:
    control = _evidence("control", [1.0])
    treatment = deepcopy(_evidence("treatment", [1.0]))
    control["fixed_invariants"]["agent_name"] = "codex"
    control["fixed_invariants"]["agent_version"] = "1.2.3"
    treatment["fixed_invariants"]["agent_name"] = "claude"
    treatment["fixed_invariants"]["agent_version"] = "1.2.3"

    report = compare_evidence(control, treatment)

    assert report["status"] == "INVALID"
    assert "fixed invariants differ" in report["validation_failures"]


def test_observation_normalization_binds_repetitions_and_model(
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
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(job), encoding="utf-8")
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
                "task_name": "jacobian/graph-counterexample",
                "task_checksum": "sha256:" + "a" * 64,
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
                    "rewards": {"correctness": 1.0, "false_certification": 0.0}
                },
                "exception_info": None,
            }
        ],
    }
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")

    evidence, failures = build_observation_evidence(
        dataset="agent-workflow-v1",
        condition="control",
        job_path=job_path,
        jobs_dir=tmp_path,
        result_path=result_path,
    )

    assert failures == []
    assert evidence["status"] == "VALID"
    assert evidence["fixed_invariants"]["model"] == "model"


def test_observation_binds_agent_and_provider_identity_and_detects_drift(
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
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(job), encoding="utf-8")
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
                "task_name": "jacobian/graph-counterexample",
                "task_checksum": "sha256:" + "a" * 64,
                "trial_name": "attempt-0",
                "agent_info": {
                    "name": "codex",
                    "version": "1.2.3",
                    "model_info": {"name": "model", "provider": "openai"},
                },
                "agent_result": {
                    "n_input_tokens": 10,
                    "n_output_tokens": 5,
                    "cost_usd": 0.01,
                },
                "verifier_result": {
                    "rewards": {"correctness": 1.0, "false_certification": 0.0}
                },
                "exception_info": None,
            }
        ],
    }
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    runtime = {"model": "model", "agent": {"name": "codex", "version": "1.2.3"}}

    evidence, failures = build_observation_evidence(
        dataset="agent-workflow-v1",
        condition="control",
        job_path=job_path,
        jobs_dir=tmp_path,
        result_path=result_path,
        runtime_snapshot=runtime,
    )

    assert failures == []
    assert evidence["fixed_invariants"]["agent_name"] == "codex"
    assert evidence["fixed_invariants"]["agent_version"] == "1.2.3"
    assert evidence["fixed_invariants"]["model_provider"] == "openai"

    # A runtime identity drift (observed agent version differs from the frozen
    # snapshot) must fail closed instead of producing a VALID comparison.
    drifted = json.loads(result_path.read_text())
    drifted["trial_results"][0]["agent_info"]["version"] = "9.9.9"
    result_path.write_text(json.dumps(drifted), encoding="utf-8")

    drifted_evidence, drifted_failures = build_observation_evidence(
        dataset="agent-workflow-v1",
        condition="control",
        job_path=job_path,
        jobs_dir=tmp_path,
        result_path=result_path,
        runtime_snapshot=runtime,
    )

    assert drifted_evidence["status"] == "INCOMPLETE"
    assert any("agent version" in failure for failure in drifted_failures)
