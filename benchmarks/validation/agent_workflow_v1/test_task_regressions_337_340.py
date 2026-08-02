from __future__ import annotations

import json
from pathlib import Path

from benchmarks.validation.agent_workflow_v1 import support


def test_cyclotomic_reciprocity_rejects_missing_factor(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "cyclotomic-reciprocity-certificate", "computed"
    )
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["result"]["factors"].pop()
    support._write_json(path, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_cyclotomic_reciprocity_rejects_corrupted_multiplicity(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "cyclotomic-reciprocity-certificate", "computed"
    )
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["result"]["factors"][0]["multiplicity"] = 1
    support._write_json(path, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0
