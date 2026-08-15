"""Generic task-specific witness binding contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import _fixtures, _verifier


def test_verifier_execution_does_not_mutate_task_bundles(tmp_path: Path) -> None:
    before = _fixtures._task_tree_snapshot()

    result = _verifier._run_verifier(
        *_fixtures._prepare_case(tmp_path, _fixtures.RATIONAL_TASK, "computed")
    )

    assert result.details["correctness"] == 1.0
    assert _fixtures._task_tree_snapshot() == before


def test_generated_json_witness_fixture_starts_valid(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(
        tmp_path, "inversion-aggregate-mask-audit", "computed"
    )

    witness_path = app / "evidence" / "inversion-audit.json"
    submission = json.loads((app / "submission.json").read_text())
    assert witness_path.is_file()
    assert submission["witness"][0]["sha256"] == _fixtures._digest(witness_path)
    assert _verifier._run_verifier(task, app, logs).reward == pytest.approx(1.0)


@pytest.mark.parametrize("task_name", _fixtures.SINGLE_EVIDENCE_TASKS)
def test_verifiers_enforce_single_witness_cardinality(
    tmp_path: Path,
    task_name: str,
) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, task_name, "computed")
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["witness"].append(dict(submission["witness"][0]))
    _fixtures._write_json(submission_path, submission)

    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.reward == 0.0
