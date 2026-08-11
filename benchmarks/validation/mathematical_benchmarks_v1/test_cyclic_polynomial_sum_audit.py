from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import support

TASK = "cyclic-polynomial-sum-audit"


def _case(tmp_path: Path):
    return support._prepare_case(tmp_path, TASK, "computed")


def test_oracle_replays_complete_elimination_certificate(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    accepted = support._run_verifier(task, app, logs)
    assert accepted.details["correctness"] == 1.0
    assert accepted.reward == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("necessary_polynomial", 2), -10),
        (("proposed_evaluations", 1), "0"),
        (("excluded_branch", "product"), "-111/8"),
        (("excluded_branch", "residual"), "0"),
    ],
)
def test_rejects_corrupted_algebraic_certificates(
    tmp_path: Path, path: tuple[object, ...], replacement: object
) -> None:
    task, app, logs = _case(tmp_path)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    target = submission["result"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_rejects_unearned_verified_claim(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["claimed_assurance"] = "VERIFIED"
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected.reward == 0.0
    assert rejected.details["false_certification"] is True
