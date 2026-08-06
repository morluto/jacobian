from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import support

TASK = "topology-generation-order-audit"


def _case(tmp_path: Path):
    return support._prepare_case(tmp_path, TASK, "computed")


def test_accepts_reordered_incomparable_family_and_different_witness(
    tmp_path: Path,
) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["result"]["input_topologies"].reverse()
    submission["result"]["witness_open_set"] = 2
    support._write_json(path, submission)
    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("generated_topology", [0, 1, 2, 4, 7]),
        ("common_subtopology", [0, 1, 7]),
        ("witness_open_set", 7),
        ("input_topologies", [[0, 7], [0, 1, 6, 7]]),
    ],
)
def test_rejects_corrupted_or_trivial_models(
    tmp_path: Path, field: str, value: object
) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["result"][field] = value
    support._write_json(path, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_rejects_unearned_verified_claim(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["claimed_assurance"] = "VERIFIED"
    support._write_json(path, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["reward"] == 0.0
    assert rejected["false_certification"] is True
