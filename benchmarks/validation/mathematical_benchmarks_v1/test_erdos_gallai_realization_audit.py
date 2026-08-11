from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import support
from jsonschema import Draft202012Validator

TASK = "erdos-gallai-realization-audit"


def test_accepts_reversed_one_based_edges(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    case = next(
        item for item in submission["result"]["cases"] if item["status"] == "GRAPHICAL"
    )
    case["edges"] = [[v + 1, u + 1] for u, v in case["edges"]]
    support._write_json(app / "submission.json", submission)
    accepted = support._run_verifier(task, app, logs)
    assert accepted.details["correctness"] == 1.0
    assert accepted.reward == pytest.approx(1.0)


def test_reference_solution_is_schema_valid() -> None:
    task = support._task(TASK)
    schema = json.loads((task / "environment" / "submission_schema.json").read_text())
    submission = json.loads((task / "solution" / "submission.json").read_text())
    Draft202012Validator(schema).validate(submission)
