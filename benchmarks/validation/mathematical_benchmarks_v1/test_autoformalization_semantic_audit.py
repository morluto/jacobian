from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import (
    _fixtures,
    _verifier,
)

TASK = "autoformalization-semantic-audit"


def test_accepts_alternative_exact_certificates(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["missing_premise_certificate"] = {
        "dimension": 1,
        "x": [-7],
        "forced_y": [0],
    }
    submission["result"]["operator_mismatch_certificate"] = {
        "dimension": 2,
        "x": [3, -2],
        "y": [2, 3],
        "dot_product": 0,
        "coordinate_products": [6, -6],
    }
    _fixtures._write_json(app / "submission.json", submission)

    accepted = _verifier._run_verifier(task, app, logs)
    assert accepted.details["correctness"] == 1.0
    assert accepted.reward == pytest.approx(1.0)


def test_rejects_incomplete_defect_set(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["defects"] = ["MISSING_DIMENSION_PREMISE"]
    _fixtures._write_json(app / "submission.json", submission)

    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_rejects_corrupted_operator_mismatch(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["operator_mismatch_certificate"]["dot_product"] = 1
    _fixtures._write_json(app / "submission.json", submission)
    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0
