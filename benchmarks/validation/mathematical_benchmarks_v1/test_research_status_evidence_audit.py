from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import support

TASK = "research-status-evidence-audit"


def _case(tmp_path: Path):
    return support._prepare_case(tmp_path, TASK, "computed")


def _rewrite(app: Path, submission: dict) -> None:
    support._bind_result_evidence(app, submission)
    support._write_json(app / "submission.json", submission)


def test_accepts_equivalent_case_order(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["cases"].reverse()
    _rewrite(app, submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


def test_rejects_wrong_scope_with_soft_scope_term(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["scope"] = "wrong scope"
    support._write_json(app / "submission.json", submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 1.0
    assert rejected["scope_accuracy"] == 0.0
    # Evidence remains valid, so reward keeps the hard base plus assurance.
    assert rejected["reward"] == pytest.approx(0.9)


def test_rejects_escaped_evidence_with_zero_aggregate_reward(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["evidence"] = [
        {
            "path": "../answer.txt",
            "sha256": submission["evidence"][0]["sha256"],
        }
    ]
    support._write_json(app / "submission.json", submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 1.0
    assert rejected["evidence_validity"] == 0.0
    assert rejected["reward"] == 0.0


def test_rejects_wrong_assurance_with_soft_assurance_term(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["claimed_assurance"] = "UNVERIFIED"
    support._write_json(app / "submission.json", submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 1.0
    assert rejected["assurance_calibration"] == 0.0
    # Evidence remains valid, so reward keeps the hard base plus scope.
    assert rejected["reward"] == pytest.approx(0.9)


def test_rejects_tampered_frozen_input(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    (app / "input.json").write_text('{"cases":[]}')
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_rejects_extra_selected_citations(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    for case in submission["result"]["cases"]:
        if case["case_id"] == "binomial-equation-resolution":
            case["selected_evidence_ids"].append("SW1999")
    _rewrite(app, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_rejects_undeclared_result_field(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["current_status"] = "ALL_FOUR_PROBLEMS_RESOLVED"
    _rewrite(app, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_rejects_non_string_case_id(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["cases"][0]["case_id"] = ["workshop-equation-status"]
    _rewrite(app, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_rejects_escaped_semantics_in_result_marker(tmp_path: Path) -> None:
    """RESULT_JSON alone is not enough when prose supports none of the claims."""
    task, app, logs = _case(tmp_path)
    evidence_path = app / "evidence" / "answer.txt"
    submission = json.loads((app / "submission.json").read_text())
    evidence_path.write_text("This is not a problem statement.\nRESULT_JSON: {}\n")
    _rewrite(app, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 1.0
    assert rejected["evidence_validity"] == 0.0
    assert rejected["reward"] == pytest.approx(0.0)


def test_oversized_evidence_does_not_block_reward_record(tmp_path: Path) -> None:
    """The evidence byte bound fires before digest hashing; reward.json is complete."""
    task, app, logs = _case(tmp_path)
    evidence_path = app / "evidence" / "answer.txt"
    submission = json.loads((app / "submission.json").read_text())
    evidence_path.write_text("x" * 2_097_152)
    # A deliberately wrong digest proves the size gate, not the hash, rejects.
    submission["evidence"][0]["sha256"] = "sha256:" + "0" * 64
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 1.0
    assert rejected["evidence_validity"] == 0.0
    assert rejected["reward"] == pytest.approx(0.0)


def test_rejects_deeply_nested_submission(tmp_path: Path) -> None:
    """Deep nesting overflows recursion; verifier writes zero reward instead of crashing."""
    task, app, logs = _case(tmp_path)
    (app / "submission.json").write_text("[" * 12000 + "1" + "]" * 12000)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_rejects_checked_assurance_above_ceiling(tmp_path: Path) -> None:
    """CHECKED is above the COMPUTED ceiling and forces reward to zero."""
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["claimed_assurance"] = "CHECKED"
    _rewrite(app, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["reward"] == 0.0


def test_rejects_invalid_utf8_evidence(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    evidence_path = app / "evidence" / "answer.txt"
    submission = json.loads((app / "submission.json").read_text())
    evidence_path.write_bytes(
        b"\xff\xfe resolution partial-progress historical problem listing"
    )
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["evidence_validity"] == 0.0


def test_requires_exponent_range_inference(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    for case in submission["result"]["cases"]:
        if case["case_id"] == "lebesgue-nagell-progress":
            case["unsupported_inferences"] = []
    _rewrite(app, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0
