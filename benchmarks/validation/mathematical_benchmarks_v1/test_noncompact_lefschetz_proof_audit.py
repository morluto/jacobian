from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import (
    _fixtures,
    _verifier,
)

TASK = "noncompact-lefschetz-proof-audit"


def _case(tmp_path: Path):
    return _fixtures._prepare_case(tmp_path, TASK, "computed")


def _rewrite(app: Path, submission: dict) -> None:
    _fixtures._write_json(app / "submission.json", submission)


def test_accepts_equivalent_rational_and_cohomology_forms(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    counterexample = submission["result"]["counterexample"]
    counterexample["translation"] = {"numerator": 2, "denominator": 2}
    counterexample["compact_support_cohomology"].reverse()
    _rewrite(app, submission)
    accepted = _verifier._run_verifier(task, app, logs)
    assert accepted.reward == pytest.approx(1.0)


@pytest.mark.parametrize("field", ["top_degree_action", "lefschetz_number"])
def test_rejects_boolean_in_integer_fields(tmp_path: Path, field: str) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["counterexample"][field] = True
    _rewrite(app, submission)
    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0


def test_enforces_visible_translation_bounds(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["counterexample"]["translation"] = {
        "numerator": 1_000_001,
        "denominator": 1,
    }
    _rewrite(app, submission)
    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
