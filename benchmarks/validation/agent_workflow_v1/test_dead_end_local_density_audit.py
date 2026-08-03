from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.agent_workflow_v1 import support

TASK = "dead-end-local-density-audit"


def test_rejects_boolean_integer_fields(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["cases"][1]["density_numerator"] = True
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0


def test_rejects_duplicate_evidence_descriptors(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["evidence"].append(dict(submission["evidence"][0]))
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["evidence_validity"] == 0.0


def test_accepts_case_reordering(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["cases"].reverse()
    support._write_json(app / "submission.json", submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


def test_rejects_duplicate_case_id(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["cases"][1]["case_id"] = submission["result"]["cases"][0][
        "case_id"
    ]
    support._write_json(app / "submission.json", submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0
