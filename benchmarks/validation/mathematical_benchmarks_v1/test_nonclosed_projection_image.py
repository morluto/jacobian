from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

from benchmarks.validation._verifier_child import run_verifier_in_child
from benchmarks.validation.mathematical_benchmarks_v1 import _fixtures, _verifier
from jsonschema import Draft202012Validator

TASK = "nonclosed-projection-image"


def test_result_protocol(tmp_path: Path) -> None:
    _fixtures.assert_result_witness_protocol(tmp_path, TASK)


def test_rejects_string_and_oversized_rationals(tmp_path: Path) -> None:
    for scenario, value in (
        ("string", "1"),
        ("oversized", {"numerator": 1 << 1_024, "denominator": 1}),
    ):
        task, app, logs = _fixtures._prepare_case(tmp_path / scenario, TASK, "computed")
        submission_path = app / "submission.json"
        submission = json.loads(submission_path.read_text())
        submission["result"]["operator_bound"] = value
        _fixtures._write_json(submission_path, submission)
        assert _verifier._run_verifier(task, app, logs).reward == 0.0


def test_accepts_unreduced_operator_bound(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "unreduced")
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["operator_bound"] = {"numerator": 2, "denominator": 2}
    _fixtures._write_json(submission_path, submission)
    assert _verifier._run_verifier(task, app, logs).reward == 1.0


def test_coordinate_count_boundary_preserves_the_diagonal_construction(
    tmp_path: Path,
) -> None:
    for terms, accepted in [(128, True), (129, False)]:
        task, app, logs = _fixtures._prepare_case(tmp_path / str(terms), TASK, "terms")
        path = app / "submission.json"
        submission = json.loads(path.read_text())
        # y_n=1/n, w_n=1/n, x_n=1. The integral comparison gives
        # sum_{n>m} y_n^2 <= 1/m, while sum x_n^2 diverges.
        submission["result"]["limit_coordinates"] = [
            {"numerator": 1, "denominator": n} for n in range(1, terms + 1)
        ]
        submission["result"]["tail_bound"]["verification_terms"] = terms
        assert sum(Fraction(1, n * n) for n in range(13, terms + 1)) < Fraction(1, 12)
        schema = json.loads((task / "environment/submission_schema.json").read_text())
        assert Draft202012Validator(schema).is_valid(submission) is accepted
        path.write_text(json.dumps(submission))
        assert run_verifier_in_child(task=task, app=app, logs=logs).reward == float(
            accepted
        )
