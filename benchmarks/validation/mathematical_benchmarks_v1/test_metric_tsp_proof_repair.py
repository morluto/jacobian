from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import _fixtures, _verifier

TASK = "metric-tsp-proof-repair"


def test_accepts_replayed_certificate(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    accepted = _verifier._run_verifier(task, app, logs)
    assert accepted.details["correctness"] == 1.0
    assert accepted.reward == 1.0
    assert accepted.reward == pytest.approx(1.0)


def test_rejects_wrong_mathematical_result(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["result"]["weights"]["optimal"] += 1
    _fixtures._write_json(path, submission)
    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_rejects_wrong_corrected_claim(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["result"]["corrected_claim"] = "SHORTCUTTING_PRESERVES_EXACT_COST"
    _fixtures._write_json(path, submission)
    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0
