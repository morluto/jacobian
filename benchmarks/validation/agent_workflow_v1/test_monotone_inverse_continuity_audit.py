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


def test_math_correctness_independent_of_protocol(tmp_path: Path) -> None:
    """A valid countermodel with a protocol defect must still report correctness."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    # Duplicating the evidence descriptor breaks the protocol but the
    # countermodel is still mathematically valid.
    submission["evidence"].append(dict(submission["evidence"][0]))
    support._write_json(app / "submission.json", submission)

    result = support._run_verifier(task, app, logs)
    assert result["correctness"] == 1.0
    assert result["reward"] == 0.0


def test_rejects_above_ceiling_assurance_claim(tmp_path: Path) -> None:
    """A CHECKED assurance claim must fail closed, not just lose assurance credit."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["claimed_assurance"] = "CHECKED"
    support._write_json(app / "submission.json", submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["reward"] == 0.0


def test_zero_reward_for_malformed_evidence(tmp_path: Path) -> None:
    """An escaped or wrong-digest evidence descriptor must zero aggregate reward."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["evidence"][0]["sha256"] = "sha256:" + "0" * 64
    support._write_json(app / "submission.json", submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["evidence_validity"] == 0.0
    assert rejected["reward"] == 0.0
    assert rejected["correctness"] == 1.0


def test_rejects_unrelated_evidence_content(tmp_path: Path) -> None:
    """Evidence with a RESULT_JSON marker that does not match the result is invalid."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    # Change the result but leave the evidence marker unchanged.
    submission["result"]["gap_witness"] = "1/3"
    support._write_json(app / "submission.json", submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 1.0
    assert rejected["evidence_validity"] == 0.0
    assert rejected["reward"] == 0.0


def test_rejects_evidence_without_result_marker(tmp_path: Path) -> None:
    """Evidence without a RESULT_JSON marker must not earn evidence credit."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    answer_path = app / "evidence" / "answer.txt"
    answer_path.write_text("A derivation without a result marker.\n")
    submission = json.loads((app / "submission.json").read_text())
    submission["evidence"][0]["sha256"] = support._digest(answer_path)
    support._write_json(app / "submission.json", submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["evidence_validity"] == 0.0
    assert rejected["reward"] == 0.0


def test_rejects_marker_only_evidence_with_unrelated_prose(tmp_path: Path) -> None:
    """Matching RESULT_JSON cannot substitute for the published derivation."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    evidence_path = app / "evidence" / "answer.txt"
    submission = json.loads((app / "submission.json").read_text())
    marker = next(
        line
        for line in evidence_path.read_text().splitlines()
        if line.startswith("RESULT_JSON:")
    )
    evidence_path.write_text("Unrelated filler.\n" + marker + "\n")
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(app / "submission.json", submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 1.0
    assert rejected["evidence_validity"] == 0.0
    assert rejected["reward"] == 0.0


def test_malformed_assurance_zeroes_scope_accuracy(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["claimed_assurance"] = []
    support._write_json(app / "submission.json", submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["scope_accuracy"] == 0.0
    assert rejected["assurance_calibration"] == 0.0
    assert rejected["reward"] == 0.0


def test_accepts_schema_valid_long_canonical_rational(tmp_path: Path) -> None:
    """A canonical rational longer than the former hidden cap remains valid."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["gap_witness"] = "1/" + "1" + "0" * 80
    support._bind_result_evidence(app, submission)
    support._write_json(app / "submission.json", submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


def test_reports_tampered_input_separately_from_math_correctness(
    tmp_path: Path,
) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    (app / "input.json").write_text("{", encoding="utf-8")

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 1.0
    assert rejected["input_binding"] == 0.0
    assert rejected["reward"] == 0.0
