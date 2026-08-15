from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import _fixtures, _verifier

TASK = "nondifferentiable-maximum-construction"


def test_accepts_typed_rational_construction(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "typed")
    assert _verifier._run_verifier(task, app, logs).reward == pytest.approx(1.0)


def test_rejects_string_coerced_slope(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "string")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["right_slope"] = "-1/2"
    _fixtures._write_json(app / "submission.json", submission)
    assert _verifier._run_verifier(task, app, logs).reward == 0.0
