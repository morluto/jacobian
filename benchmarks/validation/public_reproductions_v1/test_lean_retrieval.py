"""Regression coverage for structured Lean retrieval results."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.public_reproductions_v1._fixtures import (
    _prepare_case,
    _write_json,
)
from benchmarks.validation.public_reproductions_v1._verifier import _run_verifier


def test_lean_retrieval_rejects_free_form_tactic_text(tmp_path: Path) -> None:
    task, app, logs = _prepare_case(tmp_path, "lean-retrieval", "free-form-tactic")
    assert _run_verifier(task, app, logs).reward == pytest.approx(1.0)

    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["candidate_tactic"] = "exact Nat.gcd_zero_right n"
    _write_json(submission_path, submission)

    rejected = _run_verifier(task, app, logs)
    assert rejected.details["correctness"] == pytest.approx(0.0)
    assert rejected.reward == pytest.approx(0.0)
