from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import (
    _fixtures,
    _verifier,
)

TASK = "divisor-minimizer-exchange-audit"


def _case(tmp_path: Path):
    return _fixtures._prepare_case(tmp_path, TASK, "computed")


def test_accepts_reordered_complete_candidate_tables(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["result"]["current_candidates"].reverse()
    submission["result"]["next_candidates"].reverse()
    _fixtures._write_json(path, submission)
    assert _verifier._run_verifier(task, app, logs).reward == pytest.approx(1.0)


def test_rejects_omitted_partition(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["result"]["current_candidates"].pop()
    _fixtures._write_json(path, submission)
    assert _verifier._run_verifier(task, app, logs).details["correctness"] == 0.0


def test_rejects_corrupt_competing_candidate(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["result"]["next_candidates"][20]["value"] += 1
    _fixtures._write_json(path, submission)
    assert _verifier._run_verifier(task, app, logs).details["correctness"] == 0.0
