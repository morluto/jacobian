from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import support

TASK = "rational-pole-vieta-audit"


def _case(tmp_path: Path):
    return support._prepare_case(tmp_path, TASK, "computed")


def test_accepts_exact_symbolic_repair(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("root_sum", "4/2010"),
        ("denominator_coefficients", [24, 0, -50, 0, 35, 0, -10, 0, -1]),
        (
            "pole_square_residuals",
            [
                {"k": 1, "residual": 0},
                {"k": 2, "residual": 4},
                {"k": 3, "residual": -6},
                {"k": 4, "residual": 24},
            ],
        ),
        (
            "cleared_polynomial_coefficients",
            [0, -48240, -50, 100500, 70, -70350, -30, 20100, 4, 2010],
        ),
    ],
)
def test_rejects_corrupted_domain_or_polynomial_certificate(
    tmp_path: Path, field: str, value: object
) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["result"][field] = value
    support._write_json(path, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0 and rejected["reward"] == 0.0


def test_rejects_unearned_verified_claim(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["claimed_assurance"] = "VERIFIED"
    support._write_json(path, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["reward"] == 0.0 and rejected["false_certification"] is True
