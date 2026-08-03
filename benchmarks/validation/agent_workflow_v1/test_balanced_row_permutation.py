from __future__ import annotations

import json
from pathlib import Path

from benchmarks.validation.agent_workflow_v1 import support

TASK = "balanced-row-permutation"


def _case(tmp_path: Path):
    return support._prepare_case(tmp_path, TASK, "computed")


def _rewrite(app: Path, submission: dict) -> None:
    support._bind_result_evidence(app, submission)
    support._write_json(app / "submission.json", submission)


def test_reference_passes(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    assert support._run_verifier(task, app, logs)["reward"] == 1.0


def test_alternative_column_order_passes(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    result = submission["result"]
    order = [5, 4, 3, 2, 1, 0]
    result["row_permutations"] = [
        [row[index] for index in order] for row in result["row_permutations"]
    ]
    result["balanced_matrix"] = [
        [row[index] for index in order] for row in result["balanced_matrix"]
    ]
    result["column_layers"] = [result["column_layers"][index] for index in order]
    _rewrite(app, submission)
    assert support._run_verifier(task, app, logs)["reward"] == 1.0


def test_reused_source_position_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["row_permutations"][0][1] = submission["result"][
        "row_permutations"
    ][0][0]
    _rewrite(app, submission)
    assert support._run_verifier(task, app, logs)["reward"] == 0.0


def test_unbalanced_column_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["column_layers"][0][0]["symbol"] = 2
    _rewrite(app, submission)
    assert support._run_verifier(task, app, logs)["reward"] == 0.0


def test_matrix_layer_mismatch_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["balanced_matrix"][3][2] = 4
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
