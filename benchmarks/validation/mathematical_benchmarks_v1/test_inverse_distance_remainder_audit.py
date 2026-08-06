from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import support

TASK = "inverse-distance-remainder-audit"


def _case(tmp_path: Path):
    return support._prepare_case(tmp_path, TASK, "computed")


def _rewrite(app: Path, submission: dict) -> None:
    support._bind_result_evidence(app, submission)
    support._write_json(app / "submission.json", submission)


def test_accepts_alternative_rational_direction(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["directional_witnesses"][0] = {
        "direction": [
            {"numerator": 4, "denominator": 5},
            {"numerator": 3, "denominator": 5},
        ],
        "quadratic_coefficient": {"numerator": 23, "denominator": 50},
        "sign": "POSITIVE",
        "normalized_residual_limit": "quadratic_coefficient",
    }
    _rewrite(app, submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (
            ("second_order_term", "dot_square_coefficient"),
            {"numerator": 1, "denominator": 1},
        ),
        (
            ("directional_witnesses", 0, "quadratic_coefficient"),
            {"numerator": 2, "denominator": 1},
        ),
        (
            ("response_audit", "defects"),
            ["CUBIC_REMAINDER_FALSE"],
        ),
    ],
)
def test_rejects_corrupted_certificates(
    tmp_path: Path,
    path: tuple[object, ...],
    replacement: object,
) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    target = submission["result"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    _rewrite(app, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_enforces_visible_rational_bounds(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    coefficient = submission["result"]["directional_witnesses"][0][
        "quadratic_coefficient"
    ]
    coefficient["numerator"] *= 1_000_001
    coefficient["denominator"] *= 1_000_001
    _rewrite(app, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
