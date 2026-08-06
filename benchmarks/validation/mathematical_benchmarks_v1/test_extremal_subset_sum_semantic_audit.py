from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import support

TASK = "extremal-subset-sum-semantic-audit"


def test_accepts_alternative_exact_certificates(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["shadowing_certificate"] = {
        "target": 1,
        "first_multiplier": 3,
        "second_multiplier": 1,
        "first_extremum": 3,
        "second_extremum": 0,
    }
    submission["result"]["predicate_certificate"]["intended_witness"] = [2, 3]
    support._bind_result_evidence(app, submission)
    support._write_json(app / "submission.json", submission)
    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


def test_rejects_same_shadow_extremum(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["shadowing_certificate"]["second_multiplier"] = 1
    submission["result"]["shadowing_certificate"]["second_extremum"] = 0
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_rejects_nonoptimal_intended_witness(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["predicate_certificate"]["intended_witness"] = [2]
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_rejects_blocker_outside_legacy_candidate(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["predicate_certificate"]["legacy_witness"] = [2, 3]
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_rejects_above_ceiling_assurance(tmp_path: Path) -> None:
    """CHECKED is above the COMPUTED ceiling and must fail closed."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["claimed_assurance"] = "CHECKED"
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["reward"] == 0.0
    assert rejected["assurance_calibration"] == 0.0
    assert rejected["evidence_validity"] == 1.0


def test_rejects_boolean_in_shadowing_certificate(tmp_path: Path) -> None:
    """JSON booleans must not satisfy integer certificate fields."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    cert = submission["result"]["shadowing_certificate"]
    cert["first_extremum"] = False  # False == 0 in Python
    cert["second_extremum"] = True  # True == 1 in Python, but expected 2
    support._bind_result_evidence(app, submission)
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_rejects_boolean_target_in_predicate_certificate(tmp_path: Path) -> None:
    """A boolean target must not pass even when it equals the expected int."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    # target is 4; a boolean can't equal 4, but test with legacy_extremum=3
    # which can't be boolean-equal either.  Instead test with a field whose
    # expected value is 0 or 1 by using an alternative certificate.
    submission["result"]["shadowing_certificate"] = {
        "target": 1,
        "first_multiplier": 0,
        "second_multiplier": 1,
        "first_extremum": False,  # False == 0
        "second_extremum": True,  # True == 1
    }
    support._bind_result_evidence(app, submission)
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_rejects_boolean_universe_entry(tmp_path: Path) -> None:
    """JSON booleans must not compare equal to exact universe integers."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["predicate_certificate"]["universe"][0] = True
    support._bind_result_evidence(app, submission)
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_rejects_affirmative_lean_verification_claim(tmp_path: Path) -> None:
    """Limitations that affirm Lean verification must be rejected."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["limitations"] = [
        "Lean parsing, elaboration, and compilation are not assessed.",
        "The Lean declaration has been verified and compiles correctly.",
    ]
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["reward"] == 0.0


def test_rejects_affirmative_asymptotic_verification_claim(tmp_path: Path) -> None:
    """Limitations that affirm the corrected asymptotic conjecture must be rejected."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["limitations"] = [
        "Lean parsing, elaboration, and compilation are not assessed.",
        "The corrected asymptotic conjecture has been verified.",
    ]
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["reward"] == 0.0
    assert rejected["limitation_accuracy"] == 0.0


def test_rejects_affirmative_asymptotic_proof_claim(tmp_path: Path) -> None:
    """Proof claims are overclaims even when they avoid the word verified."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["limitations"] = [
        "Lean parsing, elaboration, and compilation are not assessed.",
        "The corrected asymptotic conjecture is proved by the audit.",
    ]
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["reward"] == 0.0
    assert rejected["limitation_accuracy"] == 0.0


def test_deeply_nested_evidence_json_does_not_crash(tmp_path: Path) -> None:
    """A deeply nested RESULT_JSON line must not crash the verifier."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    evidence_path = app / "evidence" / "answer.txt"
    nested = "null" + ",[" * 200 + "]" * 200
    evidence_path.write_text(f"RESULT_JSON: {nested}\nshadow subset not assessed\n")
    submission = json.loads((app / "submission.json").read_text())
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(app / "submission.json", submission)
    result = support._run_verifier(task, app, logs)
    assert result["evidence_validity"] == 0.0
    # Evidence validity is a bonus, not a gate; correct math still earns 0.9.
    assert result["reward"] == pytest.approx(0.9)


def test_input_tamper_preserves_math_correctness(tmp_path: Path) -> None:
    """A tampered workspace input must not zero mathematical correctness."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    (app / "input.json").write_text("{}")
    result = support._run_verifier(task, app, logs)
    assert result["correctness"] == 1.0
    assert result["input_binding"] == 0.0
    assert result["reward"] == 0.0


def test_oversized_workspace_input_fails_closed(tmp_path: Path) -> None:
    """An oversized workspace input must fail closed without crashing."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    (app / "input.json").write_text("x" * (20 * 1024 * 1024))
    result = support._run_verifier(task, app, logs)
    assert result["input_binding"] == 0.0
    assert result["reward"] == 0.0


def test_accepts_large_digest_bound_evidence(tmp_path: Path) -> None:
    """Valid evidence has no hidden byte ceiling beyond path and digest binding."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    evidence_path = app / "evidence" / "answer.txt"
    submission = json.loads((app / "submission.json").read_text())
    evidence_path.write_text(
        "shadow subset not assessed\n"
        + ("audit " * 200_000)
        + "\nRESULT_JSON: "
        + json.dumps(submission["result"], sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(app / "submission.json", submission)
    accepted = support._run_verifier(task, app, logs)
    assert accepted["evidence_validity"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)
