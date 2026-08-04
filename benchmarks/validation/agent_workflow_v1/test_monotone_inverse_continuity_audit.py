from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.agent_workflow_v1 import support

TASK = "monotone-inverse-continuity-audit"


def test_accepts_alternative_rational_jump_countermodel(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"] = {
        "left_slope": "3/2",
        "right_slope": "1/2",
        "offset": "-1",
        "jump": "5/2",
        "left_endpoint_value": "-4",
        "left_limit": "-1",
        "right_breakpoint_value": "3/2",
        "right_endpoint_value": "5/2",
        "gap_witness": "1/4",
    }
    support._bind_result_evidence(app, submission)
    support._write_json(app / "submission.json", submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


def test_rejects_witness_outside_omitted_gap(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["gap_witness"] = "3"
    support._write_json(app / "submission.json", submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_rejects_duplicate_evidence_descriptor(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["evidence"].append(dict(submission["evidence"][0]))
    support._write_json(app / "submission.json", submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["evidence_validity"] == 0.0
    assert rejected["reward"] < 1.0
