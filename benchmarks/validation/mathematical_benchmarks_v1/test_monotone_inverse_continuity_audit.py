from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import _fixtures, _verifier

TASK = "monotone-inverse-continuity-audit"


def test_accepts_replayed_countermodel(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    accepted = _verifier._run_verifier(task, app, logs)
    assert accepted.details["correctness"] == 1.0
    assert accepted.reward == 1.0
    assert accepted.reward == pytest.approx(1.0)


def test_rejects_wrong_countermodel(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["result"]["gap_witness"] = {"numerator": 0, "denominator": 1}
    _fixtures._write_json(path, submission)
    assert _verifier._run_verifier(task, app, logs).reward == 0.0


def test_rejects_string_coerced_parameter(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "string")
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["result"]["jump"] = "2"
    _fixtures._write_json(path, submission)
    assert _verifier._run_verifier(task, app, logs).reward == 0.0
