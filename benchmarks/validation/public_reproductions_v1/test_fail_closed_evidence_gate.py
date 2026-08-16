"""Golden fail-closed tasks: an undeclared witness key must zero reward."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.public_reproductions_v1._fixtures import (
    _prepare_case,
    _write_json,
)
from benchmarks.validation.public_reproductions_v1._verifier import _run_verifier

_GOLDEN_RESULT_ONLY = (
    "closed-set-distance-strengthening-audit",
    "superposition-proof-replay",
)


@pytest.mark.parametrize("task_name", _GOLDEN_RESULT_ONLY)
def test_undeclared_witness_key_zeros_reward(
    tmp_path: Path,
    task_name: str,
) -> None:
    task, app, logs = _prepare_case(tmp_path, task_name, "computed")
    accepted = _run_verifier(task, app, logs)
    assert accepted.reward == pytest.approx(1.0)

    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    assert "witness" not in submission
    submission["witness"] = []
    _write_json(submission_path, submission)

    rejected = _run_verifier(task, app, logs)
    assert rejected.reward == pytest.approx(0.0)
    if "correctness" in rejected.details:
        assert rejected.details["correctness"] in {0.0, 1.0}
