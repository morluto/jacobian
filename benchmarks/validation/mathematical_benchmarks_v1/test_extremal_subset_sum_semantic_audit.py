from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import (
    _fixtures,
    _verifier,
)

TASK = "extremal-subset-sum-semantic-audit"


def test_accepts_alternative_exact_certificates(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["shadowing_certificate"] = {
        "target": 1,
        "first_multiplier": 3,
        "second_multiplier": 1,
        "first_extremum": 3,
        "second_extremum": 0,
    }
    submission["result"]["predicate_certificate"]["intended_witness"] = [2, 3]
    _fixtures._write_json(app / "submission.json", submission)
    accepted = _verifier._run_verifier(task, app, logs)
    assert accepted.details["correctness"] == 1.0
    assert accepted.reward == pytest.approx(1.0)


def test_rejects_same_shadow_extremum(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["shadowing_certificate"]["second_multiplier"] = 1
    submission["result"]["shadowing_certificate"]["second_extremum"] = 0
    _fixtures._write_json(app / "submission.json", submission)
    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_rejects_blocker_outside_legacy_candidate(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["predicate_certificate"]["legacy_witness"] = [2, 3]
    _fixtures._write_json(app / "submission.json", submission)
    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_rejects_boolean_in_shadowing_certificate(tmp_path: Path) -> None:
    """JSON booleans must not satisfy integer certificate fields."""
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    cert = submission["result"]["shadowing_certificate"]
    cert["first_extremum"] = False  # False == 0 in Python
    cert["second_extremum"] = True  # True == 1 in Python, but expected 2
    _fixtures._write_json(app / "submission.json", submission)
    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_rejects_boolean_target_in_predicate_certificate(tmp_path: Path) -> None:
    """A boolean target must not pass even when it equals the expected int."""
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["shadowing_certificate"] = {
        "target": 1,
        "first_multiplier": 0,
        "second_multiplier": 1,
        "first_extremum": False,  # False == 0
        "second_extremum": True,  # True == 1
    }
    _fixtures._write_json(app / "submission.json", submission)
    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_rejects_boolean_universe_entry(tmp_path: Path) -> None:
    """JSON booleans must not compare equal to exact universe integers."""
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["predicate_certificate"]["universe"][0] = True
    _fixtures._write_json(app / "submission.json", submission)
    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_input_tamper_preserves_math_correctness(tmp_path: Path) -> None:
    """A tampered workspace input must not zero mathematical correctness."""
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    (app / "input.json").write_text("{}")
    result = _verifier._run_verifier(task, app, logs)
    assert result.details["correctness"] == 0.0
    assert result.reward == 0.0


def test_oversized_workspace_input_fails_closed(tmp_path: Path) -> None:
    """An oversized workspace input must fail closed without crashing."""
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    (app / "input.json").write_text("x" * (20 * 1024 * 1024))
    result = _verifier._run_verifier(task, app, logs)
    assert result.reward == 0.0
