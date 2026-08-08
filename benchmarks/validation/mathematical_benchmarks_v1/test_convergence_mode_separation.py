from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import support

_TASK = "convergence-mode-separation"


def test_rejects_unbounded_research_status_fact(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, _TASK, "computed")
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["research_scope"]["underlying_problem"] = "ADJUDICATED"
    support._bind_result_evidence(app, submission)
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)

    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == pytest.approx(0.0)


def test_result_marker_carries_semantics_without_keyword_inference(
    tmp_path: Path,
) -> None:
    task, app, logs = support._prepare_case(tmp_path, _TASK, "computed")
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    (app / "evidence" / "answer.txt").write_text(
        "Evidence is bound below.\nRESULT_JSON: {}\n"
    )
    support._bind_result_evidence(app, submission)
    support._write_json(submission_path, submission)

    accepted = support._run_verifier(task, app, logs)

    assert accepted.details["correctness"] == 1.0
    assert accepted.details["evidence_validity"] == 1.0
    assert accepted.reward == pytest.approx(1.0)


def test_result_requires_checked_structural_convergence_arguments(
    tmp_path: Path,
) -> None:
    task, app, logs = support._prepare_case(tmp_path, _TASK, "computed")
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["pointwise_argument"] = {
        "hit_count_per_level": 1,
        "miss_count_per_level": "UNSPECIFIED",
    }
    support._bind_result_evidence(app, submission)
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)

    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == pytest.approx(0.0)
