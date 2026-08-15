from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import (
    _fixtures,
    _verifier,
)

TASK = "topology-generation-order-audit"


def _case(tmp_path: Path):
    return _fixtures._prepare_case(tmp_path, TASK, "computed")


def test_accepts_reordered_incomparable_family_and_different_witness(
    tmp_path: Path,
) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["result"]["input_topologies"].reverse()
    submission["result"]["witness_open_set"] = 2
    _fixtures._write_json(path, submission)
    accepted = _verifier._run_verifier(task, app, logs)
    assert accepted.details["correctness"] == 1.0
    assert accepted.reward == pytest.approx(1.0)


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
    _fixtures._write_json(path, submission)
    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0
