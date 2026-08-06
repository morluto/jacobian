from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import support

TASK = "divisor-minimizer-exchange-audit"


def _case(tmp_path: Path):
    return support._prepare_case(tmp_path, TASK, "computed")


def test_accepts_reordered_complete_candidate_tables(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["result"]["current_candidates"].reverse()
    submission["result"]["next_candidates"].reverse()
    support._write_json(path, submission)
    assert support._run_verifier(task, app, logs)["reward"] == pytest.approx(1.0)


def test_rejects_omitted_partition(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["result"]["current_candidates"].pop()
    support._write_json(path, submission)
    assert support._run_verifier(task, app, logs)["correctness"] == 0.0


def test_rejects_corrupt_competing_candidate(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["result"]["next_candidates"][20]["value"] += 1
    support._write_json(path, submission)
    assert support._run_verifier(task, app, logs)["correctness"] == 0.0


def test_rejects_unearned_verified_claim(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["claimed_assurance"] = "VERIFIED"
    support._write_json(path, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["reward"] == 0.0
    assert rejected["false_certification"] is True
