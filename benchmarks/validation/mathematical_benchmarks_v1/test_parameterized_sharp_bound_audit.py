from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import (
    _fixtures,
    _verifier,
)

TASK = "parameterized-sharp-bound-audit"


def _case(tmp_path: Path):
    return _fixtures._prepare_case(tmp_path, TASK, "computed")


def _rewrite(app: Path, submission: dict) -> None:
    _fixtures._write_json(app / "submission.json", submission)


def test_accepts_permuted_certificates(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    result = submission["result"]
    result["certificate"]["tangent_variables"] = ["c", "a", "b"]
    result["certificate"]["schur_ordering"] = ["b", "c", "a"]
    result["boundary_family"] = {
        "vanishing_variable": "a",
        "other_variables": ["c", "b"],
        "parameter": "t->0+",
        "limit": "1/4",
        "attained_for_positive_parameter": False,
    }
    result["audit"]["defects"].reverse()
    _rewrite(app, submission)

    accepted = _verifier._run_verifier(task, app, logs)
    assert accepted.details["correctness"] == 1.0
    assert accepted.reward == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("transition", {"numerator": 4, "denominator": 1}),
        (
            "high_regime",
            {
                "condition": "d>=15/4",
                "bound": "1/4",
                "remainder_coefficient": "d-15/4",
                "attainment": "ATTAINED_FOR_ALL_D",
            },
        ),
        (
            "boundary_family",
            {
                "vanishing_variable": "c",
                "other_variables": ["a", "b"],
                "parameter": "t->0+",
                "limit": "1/4",
                "attained_for_positive_parameter": True,
            },
        ),
    ],
)
def test_rejects_corrupted_sharpness(
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
