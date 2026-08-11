from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import support

TASK = "symbolic-block-determinant-decomposition"


def _case(tmp_path: Path):
    return support._prepare_case(tmp_path, TASK, "computed")


def _rewrite(app: Path, submission: dict) -> None:
    support._bind_result_evidence(app, submission)
    support._write_json(app / "submission.json", submission)


def test_accepts_alternative_sum_zero_basis(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["basis_change"] = [
        ["1", "1", "0"],
        ["1", "-1", "1"],
        ["1", "0", "-1"],
    ]
    submission["result"]["basis_change_inverse"] = [
        ["1/3", "1/3", "1/3"],
        ["2/3", "-1/3", "-1/3"],
        ["1/3", "1/3", "-2/3"],
    ]
    _rewrite(app, submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted.details["correctness"] == 1.0
    assert accepted.reward == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("basis_change_inverse", [["1", "0", "0"], ["0", "1", "0"], ["0", "0", "1"]]),
        ("channels", ["A-B", "A+2B", "A-B"]),
        ("basis_change", [["1", "1", "1"], ["1", "-1", "0"], ["0", "0", "-1"]]),
    ],
)
def test_rejects_corrupted_certificates(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"][field] = replacement
    _rewrite(app, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_enforces_common_channel_first(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    result = submission["result"]
    for row in result["basis_change"]:
        row[0], row[1] = row[1], row[0]
    result["basis_change_inverse"][0], result["basis_change_inverse"][1] = (
        result["basis_change_inverse"][1],
        result["basis_change_inverse"][0],
    )
    result["channels"] = ["A-B", "A+2B", "A-B"]
    _rewrite(app, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
