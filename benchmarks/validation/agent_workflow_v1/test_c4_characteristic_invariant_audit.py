from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.agent_workflow_v1 import support

TASK = "c4-characteristic-invariant-audit"


def test_oracle_passes(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    accepted = support._run_verifier(task, app, logs)
    assert accepted["reward"] == pytest.approx(1.0)


def test_rejects_induced_count_corruption(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["witnesses"][1]["induced_c4_count"] = 2
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0 and rejected["reward"] == 0.0


def test_rejects_chorded_graph_claimed_induced(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["witnesses"][2]["induced_c4_count"] = 1
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0 and rejected["reward"] == 0.0


def test_rejects_unsorted_edges(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["witnesses"][0]["edges"].reverse()
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0 and rejected["reward"] == 0.0
