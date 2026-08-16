from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import (
    _fixtures,
    _verifier,
)

TASK = "symbolic-block-determinant-decomposition"


def _q(value) -> dict[str, int]:
    parsed = Fraction(value)
    return {"numerator": parsed.numerator, "denominator": parsed.denominator}


def _matrix(rows: list[list[str]]) -> list[list[dict[str, int]]]:
    return [[_q(entry) for entry in row] for row in rows]


def _case(tmp_path: Path):
    return _fixtures._prepare_case(tmp_path, TASK, "computed")


def _rewrite(app: Path, submission: dict) -> None:
    _fixtures._write_json(app / "submission.json", submission)


def test_accepts_alternative_sum_zero_basis(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["basis_change"] = _matrix(
        [
            ["1", "1", "0"],
            ["1", "-1", "1"],
            ["1", "0", "-1"],
        ]
    )
    submission["result"]["basis_change_inverse"] = _matrix(
        [
            ["1/3", "1/3", "1/3"],
            ["2/3", "-1/3", "-1/3"],
            ["1/3", "1/3", "-2/3"],
        ]
    )
    _rewrite(app, submission)

    accepted = _verifier._run_verifier(task, app, logs)
    assert accepted.details["correctness"] == 1.0
    assert accepted.reward == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        (
            "basis_change_inverse",
            _matrix([["1", "0", "0"], ["0", "1", "0"], ["0", "0", "1"]]),
        ),
        ("channels", ["A-B", "A+2B", "A-B"]),
        (
            "basis_change",
            _matrix([["1", "1", "1"], ["1", "-1", "0"], ["0", "0", "-1"]]),
        ),
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

    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_reordered_determinant_factors_are_accepted(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["determinant_identity"]["factors"] = [
        {"matrix": "A+2B", "exponent": 1},
        {"matrix": "A-B", "exponent": 2},
    ]
    _rewrite(app, submission)
    accepted = _verifier._run_verifier(task, app, logs)
    assert accepted.details["correctness"] == 1.0
    assert accepted.reward == pytest.approx(1.0)


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
    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
