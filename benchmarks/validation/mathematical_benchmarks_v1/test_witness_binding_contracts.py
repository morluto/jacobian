"""Generic task-specific witness binding contract tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import _fixtures, _verifier


def test_verifier_execution_does_not_mutate_task_bundles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Other host workers may import unrelated verifiers concurrently. Observe
    # the exact task under test in its own tree, including any new bytecode.
    isolated_tasks = tmp_path / "task-bundles"
    shutil.copytree(
        _fixtures._task(_fixtures.RATIONAL_TASK),
        isolated_tasks / _fixtures.RATIONAL_TASK,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    monkeypatch.setattr(_fixtures, "TASKS", isolated_tasks)
    before = _fixtures._task_tree_snapshot()

    result = _verifier._run_verifier(
        *_fixtures._prepare_case(tmp_path, _fixtures.RATIONAL_TASK, "computed")
    )

    assert result.details["correctness"] == 1.0
    assert _fixtures._task_tree_snapshot() == before


def test_generated_result_only_fixture_starts_valid(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(
        tmp_path, "rational-linear-solution", "computed"
    )

    submission = json.loads((app / "submission.json").read_text())
    assert "witness" not in submission
    assert _verifier._run_verifier(task, app, logs).reward == pytest.approx(1.0)


@pytest.mark.parametrize(
    "task_name",
    _fixtures.SINGLE_EVIDENCE_TASKS
    or [
        pytest.param(
            None,
            marks=pytest.mark.skip(reason="no remaining file-witness tasks"),
        )
    ],
)
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
