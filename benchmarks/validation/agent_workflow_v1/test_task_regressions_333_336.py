from __future__ import annotations

import json
from pathlib import Path

from benchmarks.validation.agent_workflow_v1 import support


def test_symmetric_divisibility_rejects_corrupted_multiplier(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "symmetric-polynomial-divisibility", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["generator_multipliers"][0][0]["coefficient"] = "2"
    support._write_json(submission_path, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_symmetric_divisibility_rejects_incomplete_multiplier(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "symmetric-polynomial-divisibility", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["generator_multipliers"][1].pop()
    support._write_json(submission_path, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0
