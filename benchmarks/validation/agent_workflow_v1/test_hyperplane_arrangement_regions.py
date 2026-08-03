from __future__ import annotations

import json
from pathlib import Path

from benchmarks.validation.agent_workflow_v1 import support

TASK = "hyperplane-arrangement-regions"


def _case(tmp_path: Path):
    return support._prepare_case(tmp_path, TASK, "computed")


def _rewrite(app: Path, submission: dict) -> None:
    support._bind_result_evidence(app, submission)
    support._write_json(app / "submission.json", submission)


def test_reference_passes(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    assert support._run_verifier(task, app, logs)["reward"] == 1.0


def test_alternative_order_and_scaling_pass(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    planes = submission["result"]["ordered_planes"]
    planes.reverse()
    for plane in planes:
        plane["coefficients"] = [3 * value for value in plane["coefficients"]]
    for item, increment in zip(planes, [1, 2, 4, 7, 0, 6, 9, 6, 12, 16], strict=True):
        item["increment"] = increment
    _rewrite(app, submission)
    assert support._run_verifier(task, app, logs)["reward"] == 1.0


def test_generic_position_increment_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["ordered_planes"][-1]["increment"] = 22
    _rewrite(app, submission)
    assert support._run_verifier(task, app, logs)["reward"] == 0.0


def test_missing_duplicate_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["duplicate_groups"] = []
    _rewrite(app, submission)
    assert support._run_verifier(task, app, logs)["reward"] == 0.0


def test_corrupted_plane_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["ordered_planes"][8]["coefficients"][3] = 2
    _rewrite(app, submission)
    assert support._run_verifier(task, app, logs)["reward"] == 0.0


def test_false_verified_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["claimed_assurance"] = "VERIFIED"
    support._write_json(app / "submission.json", submission)
    result = support._run_verifier(task, app, logs)
    assert result["reward"] == 0.0
    assert result["false_certification"] is True
