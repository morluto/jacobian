from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import support

TASK = "elementwise-fixed-no-global-invariant"


def test_accepts_equivalent_limitation_wording(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["limitations"] = [
        "This finite counterexample does not prove a general classification result."
    ]
    support._write_json(app / "submission.json", submission)
    accepted = support._run_verifier(task, app, logs)
    assert accepted["limitation_accuracy"] == 1.0
    assert accepted["reward"] == 1.0


def test_oracle_certificate_is_accepted(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == 1.0


def test_plain_digest_bound_evidence_needs_no_private_marker(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    evidence = app / "evidence/answer.txt"
    evidence.write_text(
        "Each element of the constructed group fixes a nonzero vector, "
        "but no common nonzero vector is fixed by all elements. "
        "This counterexample shows the quantifier order matters: "
        "the universal-existential statement does not imply the "
        "existential-universal one.\n"
    )
    submission["evidence"][0]["sha256"] = support._digest(evidence)
    support._write_json(app / "submission.json", submission)
    accepted = support._run_verifier(task, app, logs)
    assert accepted["evidence_validity"] == 1.0
    assert accepted["reward"] == 1.0


def test_visible_input_tamper_preserves_math_diagnostic(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    (app / "input.json").write_text("{}")
    rejected = support._run_verifier(task, app, logs)
    assert rejected["input_binding"] == 0.0
    assert rejected["correctness"] == 1.0
    assert rejected["reward"] == 0.0


def test_malformed_assurance_preserves_other_diagnostics(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["claimed_assurance"] = []
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["protocol_compliance"] == 0.0
    assert rejected["correctness"] == 1.0
    assert rejected["evidence_validity"] == 1.0
    assert rejected["scope_accuracy"] == 1.0
    assert rejected["assurance_calibration"] == 0.0
    assert rejected["reward"] == 0.0


def test_false_limitation_is_rejected(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["limitations"] = ["Proof-assistant verified complete classification."]
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["limitation_accuracy"] == 0.0
    assert rejected["reward"] == 0.0


def test_accepts_alternative_prime_field(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["field_prime"] = 7

    def replace_minus_one(value):
        if isinstance(value, list):
            return [replace_minus_one(item) for item in value]
        return 6 if value == 4 else value

    for key in ("generators", "group_elements", "fixed_vectors"):
        submission["result"][key] = replace_minus_one(submission["result"][key])
    support._bind_result_evidence(app, submission)
    support._write_json(app / "submission.json", submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


def test_rejects_corrupted_elementwise_fixed_vector(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["fixed_vectors"][0] = [1, 0, 0]
    support._bind_result_evidence(app, submission)
    support._write_json(app / "submission.json", submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_rejects_incomplete_group_closure(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["group_elements"].pop()
    submission["result"]["fixed_vectors"].pop()
    support._bind_result_evidence(app, submission)
    support._write_json(app / "submission.json", submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def _write_valid_prose_evidence(app: Path, submission: dict) -> None:
    """Write a valid quantifier-failure explanation and rebind the digest."""
    evidence = app / "evidence" / "answer.txt"
    evidence.write_text(
        "Each element of the group fixes a nonzero vector, "
        "yet no common nonzero vector is fixed by all elements. "
        "This counterexample refutes the quantifier implication: "
        "the universal-existential order does not imply the "
        "existential-universal order.\n"
    )
    submission["evidence"][0]["sha256"] = support._digest(evidence)


def test_empty_evidence_text_is_rejected(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    evidence = app / "evidence" / "answer.txt"
    evidence.write_text("")
    submission["evidence"][0]["sha256"] = support._digest(evidence)
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["evidence_validity"] == 0.0
    assert rejected["reward"] == 0.0


def test_unrelated_evidence_text_is_rejected(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    evidence = app / "evidence" / "answer.txt"
    evidence.write_text(
        "Lorem ipsum dolor sit amet consectetur adipiscing elit "
        "sed do eiusmod tempor incididunt ut labore et dolore magna.\n"
    )
    submission["evidence"][0]["sha256"] = support._digest(evidence)
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["evidence_validity"] == 0.0
    assert rejected["reward"] == 0.0


def test_evidence_missing_common_concept_is_rejected(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    evidence = app / "evidence" / "answer.txt"
    evidence.write_text(
        "Each element of the group fixes a nonzero vector. "
        "This is a counterexample to the quantifier implication.\n"
    )
    submission["evidence"][0]["sha256"] = support._digest(evidence)
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["evidence_validity"] == 0.0
    assert rejected["reward"] == 0.0


def test_evidence_missing_failure_concept_is_rejected(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    evidence = app / "evidence" / "answer.txt"
    evidence.write_text(
        "Each element fixes a nonzero vector, "
        "but no common nonzero vector is fixed by all elements.\n"
    )
    submission["evidence"][0]["sha256"] = support._digest(evidence)
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["evidence_validity"] == 0.0
    assert rejected["reward"] == 0.0


def test_evidence_contradiction_negating_elementwise_is_rejected(
    tmp_path: Path,
) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    evidence = app / "evidence" / "answer.txt"
    evidence.write_text(
        "It is false that every element fixes a vector; "
        "the common fixed claim is not verified. "
        "This is a counterexample to the quantifier implication.\n"
    )
    submission["evidence"][0]["sha256"] = support._digest(evidence)
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["evidence_validity"] == 0.0
    assert rejected["reward"] == 0.0


def test_equivalent_wording_with_not_every_is_accepted(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    evidence = app / "evidence" / "answer.txt"
    evidence.write_text(
        "Not every element fixes the same vector, but every group element "
        "does preserve some nonzero vector. There is no shared nonzero fixed "
        "vector. Thus the quantifier order is separated and the first claim "
        "does not imply the second.\n"
    )
    submission["evidence"][0]["sha256"] = support._digest(evidence)
    support._write_json(app / "submission.json", submission)
    accepted = support._run_verifier(task, app, logs)
    assert accepted["evidence_validity"] == 1.0
    assert accepted["reward"] == 1.0


def test_boolean_common_fixed_dimension_preserves_correctness_diagnostic(
    tmp_path: Path,
) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["common_fixed_dimension"] = False
    support._bind_result_evidence(app, submission)
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_unverified_assurance_is_accepted(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["claimed_assurance"] = "UNVERIFIED"
    _write_valid_prose_evidence(app, submission)
    support._write_json(app / "submission.json", submission)
    accepted = support._run_verifier(task, app, logs)
    assert accepted["assurance_calibration"] == 1.0
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == 1.0


def test_large_valid_evidence_is_accepted(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    evidence = app / "evidence" / "answer.txt"
    base = (
        "Each element of the group fixes a nonzero vector, "
        "yet no common nonzero vector is fixed by all elements. "
        "This counterexample refutes the quantifier implication: "
        "the universal-existential order does not imply the "
        "existential-universal order.\n"
    )
    evidence.write_text(base * 500)
    submission["evidence"][0]["sha256"] = support._digest(evidence)
    support._write_json(app / "submission.json", submission)
    accepted = support._run_verifier(task, app, logs)
    assert accepted["evidence_validity"] == 1.0
    assert accepted["reward"] == 1.0
