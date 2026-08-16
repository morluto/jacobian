from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import (
    _fixtures,
    _verifier,
)

TASK = "generalized-shift-proof-audit"


def load_case(tmp_path: Path):
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission_path = app / "submission.json"
    return task, app, logs, submission_path, json.loads(submission_path.read_text())


def write_result(submission_path: Path, submission: dict) -> None:
    _fixtures._write_json(submission_path, {"result": submission["result"]})


def test_canonical_certificate_receives_full_reward(tmp_path: Path) -> None:
    task, app, logs, _, _ = load_case(tmp_path)
    result = _verifier._run_verifier(task, app, logs)
    assert result.reward == pytest.approx(1.0)


def test_alternative_exact_certificates_are_accepted(tmp_path: Path) -> None:
    task, app, logs, submission_path, submission = load_case(tmp_path)
    submission["result"] = {
        "collision": {
            "first_index": -2,
            "second_index": 3,
            "alpha_first": 5,
            "alpha_second": 0,
        },
        "fourier_block": {
            "size": 9,
            "operator_norm_squared": {"numerator": 9, "denominator": 4},
        },
        "norm_direction": {
            "valid_relation": "OPERATOR_NORM_LE_HILBERT_SCHMIDT_NORM",
            "diagonal_entries": [
                {"numerator": -5, "denominator": 2},
                {"numerator": 1, "denominator": 3},
            ],
            "operator_norm_squared": {"numerator": 25, "denominator": 4},
            "hilbert_schmidt_norm_squared": {
                "numerator": 229,
                "denominator": 36,
            },
        },
        "radical_domain": {"m": 17, "radicand": -16, "real_status": "NOT_REAL"},
    }
    write_result(submission_path, submission)
    result = _verifier._run_verifier(task, app, logs)
    assert result.reward == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("section", "field", "bad_value"),
    [
        ("collision", "alpha_second", 1),
        ("fourier_block", "operator_norm_squared", {"numerator": 25, "denominator": 4}),
        (
            "norm_direction",
            "operator_norm_squared",
            {"numerator": 25, "denominator": 1},
        ),
        ("radical_domain", "radicand", -1),
    ],
)
def test_corrupted_defect_certificates_are_rejected(
    tmp_path: Path, section: str, field: str, bad_value: object
) -> None:
    task, app, logs, submission_path, submission = load_case(tmp_path)
    submission["result"][section][field] = bad_value
    write_result(submission_path, submission)
    result = _verifier._run_verifier(task, app, logs)
    assert result.details["correctness"] == 0.0
    assert result.reward == 0.0


def test_undeclared_nested_certificate_field_is_rejected(
    tmp_path: Path,
) -> None:
    """An extra field in a nested certificate object must be rejected."""
    task, app, logs, submission_path, submission = load_case(tmp_path)
    submission["result"]["fourier_block"]["extra_field"] = 0
    write_result(submission_path, submission)
    result = _verifier._run_verifier(task, app, logs)
    assert result.details["correctness"] == 0.0
    assert result.reward == 0.0


def test_unreduced_rationals_are_accepted(tmp_path: Path) -> None:
    """Schema-valid unreduced rationals must receive full reward."""
    task, app, logs, submission_path, submission = load_case(tmp_path)
    submission["result"]["norm_direction"]["diagonal_entries"] = [
        {"numerator": 6, "denominator": 2},
        {"numerator": 8, "denominator": 2},
    ]
    submission["result"]["norm_direction"]["operator_norm_squared"] = {
        "numerator": 32,
        "denominator": 2,
    }
    submission["result"]["norm_direction"]["hilbert_schmidt_norm_squared"] = {
        "numerator": 50,
        "denominator": 2,
    }
    write_result(submission_path, submission)
    result = _verifier._run_verifier(task, app, logs)
    assert result.reward == pytest.approx(1.0)


def test_collision_out_of_bounds_is_rejected(tmp_path: Path) -> None:
    """Collision indices outside the schema-declared [-20, 20] range must be
    rejected even though they satisfy the mathematical collision equation."""
    task, app, logs, submission_path, submission = load_case(tmp_path)
    submission["result"]["collision"] = {
        "first_index": 1000,
        "second_index": 1001,
        "alpha_first": 1,
        "alpha_second": 0,
    }
    write_result(submission_path, submission)
    result = _verifier._run_verifier(task, app, logs)
    assert result.details["correctness"] == 0.0
    assert result.reward == 0.0


def test_oversized_diagonal_entries_is_rejected(tmp_path: Path) -> None:
    """More diagonal entries than the schema-declared maxItems of 8 must be
    rejected."""
    task, app, logs, submission_path, submission = load_case(tmp_path)
    submission["result"]["norm_direction"]["diagonal_entries"] = [
        {"numerator": 1, "denominator": 1} for _ in range(9)
    ]
    submission["result"]["norm_direction"]["operator_norm_squared"] = {
        "numerator": 1,
        "denominator": 1,
    }
    submission["result"]["norm_direction"]["hilbert_schmidt_norm_squared"] = {
        "numerator": 9,
        "denominator": 1,
    }
    write_result(submission_path, submission)
    result = _verifier._run_verifier(task, app, logs)
    assert result.details["correctness"] == 0.0
    assert result.reward == 0.0


def test_oversized_submission_is_rejected(tmp_path: Path) -> None:
    """An oversized submission.json must be rejected without crashing."""
    task, app, logs, _submission_path, _submission = load_case(tmp_path)
    (app / "submission.json").write_text('{"a": 1' + ", " * (2 * 1024 * 1024) + "}")
    result = _verifier._run_verifier(task, app, logs)
    assert result.details["correctness"] == 0.0
    assert result.reward == 0.0
