from __future__ import annotations

import json
from pathlib import Path

from benchmarks.validation.public_reproductions_v1._fixtures import (
    _prepare_case,
    _write_json,
)
from benchmarks.validation.public_reproductions_v1._verifier import _run_verifier

TASK = "coin-process-potential"


def _case(tmp_path: Path):
    return _prepare_case(tmp_path, TASK, "computed")


def _rewrite(app: Path, submission: dict) -> None:
    _write_json(app / "submission.json", {"result": submission["result"]})


def test_reference_passes(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    assert _run_verifier(task, app, logs).reward == 1.0


def test_corrupted_weight_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["linear_weights"][7] += 1
    _rewrite(app, submission)
    assert _run_verifier(task, app, logs).reward == 0.0


def test_wrong_pair_coefficient_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["pair_coefficient"] = -1
    _rewrite(app, submission)
    assert _run_verifier(task, app, logs).reward == 0.0


def test_published_average_without_potential_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["linear_weights"] = [0] * 12
    submission["result"]["pair_coefficient"] = 0
    _rewrite(app, submission)
    assert _run_verifier(task, app, logs).reward == 0.0


def test_forged_layer_minimum_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["minimum_by_head_count"][9] = 8
    _rewrite(app, submission)
    assert _run_verifier(task, app, logs).reward == 0.0


def test_boolean_integer_fields_are_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["minimum_by_head_count"][0] = False
    submission["result"]["minimum_by_head_count"][1] = True
    submission["result"]["average_stopping_time"]["denominator"] = True
    _rewrite(app, submission)
    result = _run_verifier(task, app, logs)
    assert result.details["correctness"] == 0.0
    assert result.reward == 0.0


def test_equivalent_unreduced_average_is_accepted(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["average_stopping_time"] = {
        "numerator": 78,
        "denominator": 2,
    }
    _rewrite(app, submission)
    assert _run_verifier(task, app, logs).reward == 1.0


def test_unused_prose_file_does_not_affect_reward(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    (app / "evidence").mkdir(exist_ok=True)
    (app / "evidence" / "answer.txt").write_text("\n")
    assert _run_verifier(task, app, logs).reward == 1.0


def test_undeclared_witness_key_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["witness"] = [
        {"path": "evidence/answer.txt", "sha256": "sha256:" + "0" * 64}
    ]
    _write_json(app / "submission.json", submission)
    assert _run_verifier(task, app, logs).reward == 0.0


def test_result_only_submission_is_accepted(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    _rewrite(app, submission)
    assert _run_verifier(task, app, logs).reward == 1.0
