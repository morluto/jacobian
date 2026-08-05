from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.agent_workflow_v1 import support

TASK = "elementwise-fixed-no-global-invariant"


def test_accepts_alternative_prime_field(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["field_prime"] = 7

    def replace_minus_one(value):
        if isinstance(value, list):
            return [replace_minus_one(item) for item in value]
        return 6 if value == 4 else value

    for key in ("generators", "group_elements", "fixed_vectors"):
        submission["result"][key] = replace_minus_one(submission["result"][key])
    support._bind_result_evidence(app, submission)
    support._write_json(app / "submission.json", submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


def test_rejects_corrupted_elementwise_fixed_vector(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["fixed_vectors"][0] = [1, 0, 0]
    support._bind_result_evidence(app, submission)
    support._write_json(app / "submission.json", submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_rejects_incomplete_group_closure(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["group_elements"].pop()
    submission["result"]["fixed_vectors"].pop()
    support._bind_result_evidence(app, submission)
    support._write_json(app / "submission.json", submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0
