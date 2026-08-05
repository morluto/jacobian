from __future__ import annotations

import json
import os
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


def test_boolean_row_position_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["row_permutations"][0][:2] = [False, True]
    _rewrite(app, submission)
    assert support._run_verifier(task, app, logs)["reward"] == 0.0


def test_symlinked_workspace_input_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    original = app / "input-original.json"
    (app / "input.json").rename(original)
    (app / "input.json").symlink_to(original)
    assert support._run_verifier(task, app, logs)["reward"] == 0.0


def test_affirmative_general_theorem_claim_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["limitations"] = ["A proof assistant verifies the general theorem."]
    _rewrite(app, submission)
    result = support._run_verifier(task, app, logs)
    assert result["correctness"] == 0.0
    assert result["scope_accuracy"] == 0.0
    assert result["reward"] == 0.0


def test_oversized_evidence_is_rejected_without_crashing(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    (app / "evidence" / "answer.txt").write_text(
        "column row source position exactly " + "x" * 65536
    )
    submission["evidence"][0]["sha256"] = support._digest(
        app / "evidence" / "answer.txt"
    )
    support._write_json(app / "submission.json", submission)
    result = support._run_verifier(task, app, logs)
    assert result["evidence_validity"] == 0.0
    assert result["reward"] == 0.0


def test_unverified_claim_preserves_other_metric_axes(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["claimed_assurance"] = "UNVERIFIED"
    support._write_json(app / "submission.json", submission)
    result = support._run_verifier(task, app, logs)
    assert result["correctness"] == 1.0
    assert result["evidence_validity"] == 1.0
    assert result["scope_accuracy"] == 1.0
    assert result["assurance_calibration"] == 0.0
    assert result["reward"] == 0.0


def test_missing_visible_input_fails_closed(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    (app / "input.json").unlink()
    result = support._run_verifier(task, app, logs)
    assert result["correctness"] == 0.0
    assert result["input_integrity"] == 0.0
    assert result["reward"] == 0.0


def test_keyword_only_evidence_without_result_binding_is_rejected(
    tmp_path: Path,
) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    evidence = app / "evidence" / "answer.txt"
    evidence.write_text("column row source position exactly\n")
    submission["evidence"][0]["sha256"] = support._digest(evidence)
    support._write_json(app / "submission.json", submission)
    result = support._run_verifier(task, app, logs)
    assert result["evidence_validity"] == 0.0
    assert result["reward"] == 0.0


def test_boolean_in_result_json_evidence_is_rejected(tmp_path: Path) -> None:
    """RESULT_JSON must match the submitted result without bool/int coercion."""
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    evidence = app / "evidence" / "answer.txt"
    coerced = json.loads(json.dumps(submission["result"]))
    assert coerced["balanced_matrix"][0][0] == 1
    coerced["balanced_matrix"][0][0] = True
    assert coerced == submission["result"]  # Python coerces True == 1
    evidence.write_text(
        "column row source position exactly\n"
        "RESULT_JSON:"
        + json.dumps(coerced, sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    submission["evidence"][0]["sha256"] = support._digest(evidence)
    support._write_json(app / "submission.json", submission)
    result = support._run_verifier(task, app, logs)
    assert result["evidence_validity"] == 0.0
    assert result["reward"] == 0.0


def test_symlinked_submission_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    original = app / "submission-original.json"
    (app / "submission.json").rename(original)
    (app / "submission.json").symlink_to(original)
    result = support._run_verifier(task, app, logs)
    assert result["reward"] == 0.0


def test_nonregular_input_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    (app / "input.json").unlink()
    os.mkfifo(app / "input.json")
    result = support._run_verifier(task, app, logs)
    assert result["input_integrity"] == 0.0
    assert result["reward"] == 0.0
