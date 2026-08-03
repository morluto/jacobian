from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.agent_workflow_v1 import support

TASK = "propositional-rewrite-trace-replay"


def _bind_structured_certificate(
    task: Path, app: Path, submission: dict, name: str
) -> None:
    certificate = json.loads((task / "solution" / name).read_text())
    certificate["result"] = submission["result"]
    target = app / "evidence" / name
    support._write_json(target, certificate)
    submission["evidence"][0]["sha256"] = support._digest(target)


def test_propositional_rewrite_trace_accepts_oracle(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "propositional-rewrite-trace-replay", "computed"
    )
    submission = json.loads((app / "submission.json").read_text())
    _bind_structured_certificate(task, app, submission, "rewrite-trace.json")
    support._write_json(app / "submission.json", submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


def test_propositional_rewrite_trace_accepts_alternative_order(
    tmp_path: Path,
) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "propositional-rewrite-trace-replay", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    original = submission["result"]["steps"]
    after_outer = original[0]["after_ast"]
    after_double = json.loads(json.dumps(after_outer))
    after_double["args"][2] = {"op": "atom", "name": "Authoritarian(x)"}
    after_first_imp = json.loads(json.dumps(after_double))
    after_first_imp["args"][0] = original[1]["after_ast"]["args"][0]
    submission["result"]["steps"] = [
        original[0],
        {"rule": "DOUBLE_NEGATION", "path": [2], "after_ast": after_double},
        {"rule": "NOT_IMPLICATION", "path": [0], "after_ast": after_first_imp},
        {"rule": "NOT_IMPLICATION", "path": [1], "after_ast": original[3]["after_ast"]},
        *original[4:],
    ]
    _bind_structured_certificate(task, app, submission, "rewrite-trace.json")
    support._write_json(submission_path, submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


def test_propositional_rewrite_trace_rejects_equivalent_jump(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "propositional-rewrite-trace-replay", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["steps"][0]["after_ast"] = {"op": "false"}
    _bind_structured_certificate(task, app, submission, "rewrite-trace.json")
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0
