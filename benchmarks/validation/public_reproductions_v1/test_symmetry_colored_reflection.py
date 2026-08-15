"""Input-recomputation regressions for colored-reflection symmetry."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from benchmarks.validation.public_reproductions_v1._fixtures import (
    _prepare_case,
    _write_json,
)
from benchmarks.validation.public_reproductions_v1._verifier import _run_verifier


def test_symmetry_recomputes_orbits_and_rejects_nested_endpoint_bypass(
    tmp_path: Path,
) -> None:
    task, app, logs = _prepare_case(
        tmp_path, "symmetry-colored-reflection", "input-recomputation"
    )
    copied_task = tmp_path / "symmetry-task"
    shutil.copytree(task, copied_task)
    expected_path = copied_task / "tests" / "expected.json"
    expected = json.loads(expected_path.read_text())
    expected["expected_edge_orbits"] = []
    _write_json(expected_path, expected)

    assert _run_verifier(copied_task, app, logs).reward == pytest.approx(1.0)

    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["edge_orbits"] = [[[["a"], ["b"]], [["b"], ["c"]]]]
    _write_json(submission_path, submission)

    assert _run_verifier(copied_task, app, logs).reward == 0.0
