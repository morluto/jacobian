from __future__ import annotations

import json
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1 import (
    _fixtures,
    _verifier,
)

TASK = "calendar-good-days-audit"


def test_rejects_corrupted_count(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["count"] = 15
    _fixtures._write_json(app / "submission.json", submission)
    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_tampered_input_is_a_hard_gate_without_erasing_math_diagnostic(
    tmp_path: Path,
) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    input_data = json.loads((app / "input.json").read_text())
    input_data["task_id"] = "tampered"
    _fixtures._write_json(app / "input.json", input_data)

    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 1.0
    assert rejected.reward == 0.0
