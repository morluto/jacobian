from __future__ import annotations

import json
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1 import (
    _fixtures,
    _verifier,
)

TASK = "exponential-moment-rationality"


def test_uses_result_only_protocol(tmp_path: Path) -> None:
    _fixtures.assert_result_witness_protocol(tmp_path, TASK)


def test_rejects_corrupted_generic_formula(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["generic_formula"]["numerator"][0]["coefficient"] = {
        "numerator": 2,
        "denominator": 1,
    }
    _fixtures._write_json(app / "submission.json", submission)
    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_accepts_reversed_term_order(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "reversed-terms")
    submission = json.loads((app / "submission.json").read_text())
    formula = submission["result"]["generic_formula"]
    formula["numerator"] = list(reversed(formula["numerator"]))
    _fixtures._write_json(app / "submission.json", submission)
    accepted = _verifier._run_verifier(task, app, logs)
    assert accepted.details["correctness"] == 1.0
    assert accepted.reward == 1.0


def test_rejects_missing_singular_branch(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["singular_formula"] = submission["result"]["generic_formula"]
    _fixtures._write_json(app / "submission.json", submission)
    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0
