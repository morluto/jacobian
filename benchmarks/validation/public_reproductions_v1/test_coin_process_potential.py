from __future__ import annotations

import json
from pathlib import Path

from benchmarks.validation.public_reproductions_v1 import support

TASK = "coin-process-potential"


def _case(tmp_path: Path):
    return support._prepare_case(tmp_path, TASK, "computed")


def _rewrite(app: Path, submission: dict) -> None:
    support._bind_result_evidence(app, submission)
    support._write_json(app / "submission.json", submission)


def test_reference_passes(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    assert support._run_verifier(task, app, logs)["reward"] == 1.0


def test_corrupted_weight_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["linear_weights"][7] += 1
    _rewrite(app, submission)
    assert support._run_verifier(task, app, logs)["reward"] == 0.0


def test_wrong_pair_coefficient_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["pair_coefficient"] = -1
    _rewrite(app, submission)
    assert support._run_verifier(task, app, logs)["reward"] == 0.0


def test_published_average_without_potential_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["linear_weights"] = [0] * 12
    submission["result"]["pair_coefficient"] = 0
    _rewrite(app, submission)
    assert support._run_verifier(task, app, logs)["reward"] == 0.0


def test_forged_layer_minimum_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["minimum_by_head_count"][9] = 8
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


def test_boolean_integer_fields_are_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["minimum_by_head_count"][0] = False
    submission["result"]["minimum_by_head_count"][1] = True
    submission["result"]["average_stopping_time"]["denominator"] = True
    _rewrite(app, submission)
    result = support._run_verifier(task, app, logs)
    assert result["correctness"] == 0.0
    assert result["reward"] == 0.0


def test_equivalent_unreduced_average_is_accepted(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["average_stopping_time"] = {
        "numerator": 78,
        "denominator": 2,
    }
    _rewrite(app, submission)
    assert support._run_verifier(task, app, logs)["reward"] == 1.0


def test_empty_evidence_is_rejected_without_keyword_scoring(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    (app / "evidence" / "answer.txt").write_text("\n")
    _rewrite(app, submission)
    result = support._run_verifier(task, app, logs)
    assert result["correctness"] == 1.0
    assert result["evidence_validity"] == 0.0
    assert result["reward"] == 0.0


def test_conservative_assurance_preserves_other_dimensions(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["claimed_assurance"] = "UNVERIFIED"
    _rewrite(app, submission)
    result = support._run_verifier(task, app, logs)
    assert result["correctness"] == 1.0
    assert result["evidence_validity"] == 1.0
    assert result["scope_accuracy"] == 1.0
    assert result["assurance_calibration"] == 0.0
    assert result["reward"] == 0.0


def test_false_verified_preserves_evidence_and_scope(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["claimed_assurance"] = "VERIFIED"
    _rewrite(app, submission)
    result = support._run_verifier(task, app, logs)
    assert result["correctness"] == 1.0
    assert result["evidence_validity"] == 1.0
    assert result["scope_accuracy"] == 1.0
    assert result["assurance_calibration"] == 0.0
    assert result["false_certification"] is True
    assert result["reward"] == 0.0


def test_generality_claim_in_limitations_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["limitations"] = ["This proves the result for arbitrary n."]
    _rewrite(app, submission)
    result = support._run_verifier(task, app, logs)
    assert result["correctness"] == 0.0
    assert result["scope_accuracy"] == 0.0
    assert result["reward"] == 0.0


def test_fixed_scope_limitation_without_keyword_is_accepted(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["limitations"] = [
        "This certificate applies only to the frozen 12-coin instance (n=12)."
    ]
    _rewrite(app, submission)
    result = support._run_verifier(task, app, logs)
    assert result["scope_accuracy"] == 1.0
    assert result["reward"] == 1.0
