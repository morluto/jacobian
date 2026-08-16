from __future__ import annotations

import json
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1 import _fixtures, _verifier

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
