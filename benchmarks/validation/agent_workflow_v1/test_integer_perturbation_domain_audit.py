from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.agent_workflow_v1 import support

TASK = "integer-perturbation-domain-audit"


def test_oracle_passes(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    accepted = support._run_verifier(task, app, logs)
    assert accepted["reward"] == pytest.approx(1.0)


def test_accepts_alternative_periodic_witness(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    witness = submission["result"]["integer_witness"]
    witness.update(
        {
            "period": 5,
            "a_values": [3, 4, 6, 8, 9],
            "b_values": [-3, 2, -6, -1, 4],
            "sum_values": [0, 6, 0, 7, 13],
            "b_min": -6,
            "b_max": 4,
            "cancellation_indices": [0, 2],
        }
    )
    support._bind_result_evidence(app, submission)
    support._write_json(app / "submission.json", submission)
    accepted = support._run_verifier(task, app, logs)
    assert accepted["reward"] == pytest.approx(1.0)


def test_accepts_unordered_cancellation_indices(tmp_path: Path) -> None:
    """Cancellation indices are a set in the public contract, not a sequence."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["integer_witness"]["cancellation_indices"] = [2, 0]
    support._bind_result_evidence(app, submission)
    support._write_json(app / "submission.json", submission)
    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


def test_accepts_witness_without_three_distinct_values(tmp_path: Path) -> None:
    """The visible contract does not require three distinct a or b values."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    witness = submission["result"]["integer_witness"]
    witness.update(
        {
            "period": 4,
            "a_values": [1, 1, 2, 2],
            "b_values": [-1, 1, -2, 3],
            "sum_values": [0, 2, 0, 5],
            "b_min": -2,
            "b_max": 3,
            "cancellation_indices": [0, 2],
        }
    )
    support._bind_result_evidence(app, submission)
    support._write_json(app / "submission.json", submission)
    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


def test_rejects_missing_cancellation(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["integer_witness"]["cancellation_indices"] = [0]
    support._bind_result_evidence(app, submission)
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0 and rejected["reward"] == 0.0


def test_rejects_noninteger_domain_shortcut(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["integer_witness"]["b_values"][0] = 0
    support._bind_result_evidence(app, submission)
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0 and rejected["reward"] == 0.0


def test_rejects_boolean_lower_bound_certificates(tmp_path: Path) -> None:
    """JSON booleans must not satisfy integer lower-bound fields."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    redundancy = submission["result"]["nat_redundancy"]
    redundancy["a_lower_bound"] = False  # False == 0
    redundancy["b_lower_bound"] = True  # True == 1
    redundancy["sum_lower_bound"] = True  # True == 1
    support._bind_result_evidence(app, submission)
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_rejects_boolean_witness_extrema(tmp_path: Path) -> None:
    """JSON booleans must not satisfy b_min/b_max integer fields."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    witness = submission["result"]["integer_witness"]
    # Use a witness where b_min would be 0 or 1 to test boolean injection
    witness.update(
        {
            "period": 4,
            "a_values": [1, 2, 3, 4],
            "b_values": [1, -1, 2, -2],
            "sum_values": [2, 1, 5, 2],
            "b_min": False,  # False == 0, but actual min is -2
            "b_max": True,  # True == 1, but actual max is 2
            "cancellation_indices": [],
        }
    )
    support._bind_result_evidence(app, submission)
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_rejects_symlinked_evidence_directory(tmp_path: Path) -> None:
    """A symlinked evidence/ directory must not escape the workspace."""
    import shutil

    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    forged_dir = app / "forged"
    forged_dir.mkdir()
    forged_answer = forged_dir / "answer.txt"
    submission = json.loads((app / "submission.json").read_text())
    original = (app / "evidence" / "answer.txt").read_text()
    forged_answer.write_text(original)
    submission["evidence"][0]["sha256"] = support._digest(forged_answer)
    support._write_json(app / "submission.json", submission)
    shutil.rmtree(app / "evidence")
    (app / "evidence").symlink_to(forged_dir)
    result = support._run_verifier(task, app, logs)
    assert result["evidence_validity"] == 0.0
    assert result["reward"] == 0.0


def test_deeply_nested_evidence_json_does_not_crash(tmp_path: Path) -> None:
    """A deeply nested RESULT_JSON line must not crash the verifier."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    evidence_path = app / "evidence" / "answer.txt"
    nested = "null" + ",[" * 200 + "]" * 200
    evidence_path.write_text(f"RESULT_JSON: {nested}\nnatural integer not assessed\n")
    submission = json.loads((app / "submission.json").read_text())
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(app / "submission.json", submission)
    result = support._run_verifier(task, app, logs)
    assert result["evidence_validity"] == 0.0
    assert result["reward"] == 0.0


def test_rejects_assertion_style_theorem_overclaim(tmp_path: Path) -> None:
    """A theorem can be overclaimed without using proof or verification verbs."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    evidence_path = app / "evidence" / "answer.txt"
    evidence_path.write_text(
        "Nat-domain certificate and Z-valued periodic witness. "
        "No irrationality theorem is claimed, but the theorem is true.\n"
        + next(
            line
            for line in evidence_path.read_text().splitlines()
            if line.startswith("RESULT_JSON:")
        )
        + "\n"
    )
    submission = json.loads((app / "submission.json").read_text())
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["evidence_validity"] == 0.0
    assert rejected["reward"] == 0.0


def test_accepts_equivalent_concise_audit_evidence(tmp_path: Path) -> None:
    """Equivalent Nat/Z wording satisfies the public concise-audit requirement."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    evidence_path = app / "evidence" / "answer.txt"
    result_marker = next(
        line
        for line in evidence_path.read_text().splitlines()
        if line.startswith("RESULT_JSON:")
    )
    evidence_path.write_text(
        "Nat-domain certificate and Z-valued periodic witness. "
        "No Lean or theorem claim is made.\n"
        f"{result_marker}\n"
    )
    submission = json.loads((app / "submission.json").read_text())
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(app / "submission.json", submission)
    accepted = support._run_verifier(task, app, logs)
    assert accepted["evidence_validity"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


def test_rejects_multiple_result_markers(tmp_path: Path) -> None:
    """Evidence must bind exactly one unambiguous result marker."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    evidence_path = app / "evidence" / "answer.txt"
    original = evidence_path.read_text()
    marker = next(
        line for line in original.splitlines() if line.startswith("RESULT_JSON:")
    )
    evidence_path.write_text(original + marker + "\n")
    submission = json.loads((app / "submission.json").read_text())
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["evidence_validity"] == 0.0
    assert rejected["reward"] == 0.0


def test_rejects_evidence_without_result_object(tmp_path: Path) -> None:
    """A null marker cannot bind evidence when the submission has no result."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission.pop("result")
    evidence_path = app / "evidence" / "answer.txt"
    evidence_path.write_text(
        evidence_path.read_text().replace(
            next(
                line
                for line in evidence_path.read_text().splitlines()
                if line.startswith("RESULT_JSON:")
            ),
            "RESULT_JSON: null",
        )
    )
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["evidence_validity"] == 0.0
    assert rejected["protocol_compliance"] == 0.0
    assert rejected["reward"] == 0.0


def test_reports_protocol_compliance_separately(tmp_path: Path) -> None:
    """Envelope failure is visible independently and gates aggregate reward."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["task_id"] = "wrong-task"
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["protocol_compliance"] == 0.0
    assert rejected["correctness"] == 1.0
    assert rejected["evidence_validity"] == 1.0
    assert rejected["reward"] == 0.0


def test_rejects_affirmative_irrationality_claim_in_evidence(tmp_path: Path) -> None:
    """Evidence prose must obey the irrationality limitation."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    evidence_path = app / "evidence" / "answer.txt"
    evidence_path.write_text(
        "natural integer perturbations are not assessed. "
        "The source irrationality theorem has been proved.\n"
        + evidence_path.read_text()
    )
    submission = json.loads((app / "submission.json").read_text())
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["evidence_validity"] == 0.0


def test_rejects_missing_irrationality_limitation(tmp_path: Path) -> None:
    """Both excluded claims require an explicit limitation."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["limitations"] = ["Lean compilation is not assessed."]
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["limitation_accuracy"] == 0.0


def test_rejects_not_only_irrationality_overclaim(tmp_path: Path) -> None:
    """The phrase 'not only' is not a negation of the following claim."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["limitations"] = [
        "Lean compilation is not assessed. Not only has the source irrationality "
        "theorem been proved, it has been verified."
    ]
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["limitation_accuracy"] == 0.0


def test_rejects_bool_int_evidence_marker_coercion(tmp_path: Path) -> None:
    """Digest-bound JSON must preserve exact integer-versus-boolean types."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    evidence_path = app / "evidence" / "answer.txt"
    submission = json.loads((app / "submission.json").read_text())
    marker_result = json.loads(json.dumps(submission["result"]))
    marker_result["nat_redundancy"]["a_lower_bound"] = False
    evidence_path.write_text(
        "natural integer not assessed\nRESULT_JSON: "
        + json.dumps(marker_result, sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["evidence_validity"] == 0.0


def test_scope_diagnostic_is_independent_of_assurance(tmp_path: Path) -> None:
    """An unsupported assurance must not erase a correct scope diagnostic."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["claimed_assurance"] = "CHECKED"
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["scope_accuracy"] == 1.0
    assert rejected["assurance_calibration"] == 0.0


def test_rejects_affirmative_irrationality_claim(tmp_path: Path) -> None:
    """Limitations that affirm an irrationality theorem must be rejected."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["limitations"] = [
        "Lean compilation is not assessed.",
        "The irrationality theorem has been proved.",
    ]
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["reward"] == 0.0


def test_rejects_affirmative_lean_verification_claim(tmp_path: Path) -> None:
    """Limitations that affirm Lean verification must be rejected."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["limitations"] = [
        "Lean compilation is not assessed.",
        "The Lean declaration has been verified.",
    ]
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["reward"] == 0.0


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


def test_large_valid_evidence_is_accepted(tmp_path: Path) -> None:
    """No undocumented evidence size cap; large valid evidence passes."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    evidence_path = app / "evidence" / "answer.txt"
    # Build a valid evidence file larger than 1 MiB
    lines = []
    for line in evidence_path.read_text().splitlines():
        if line.startswith("RESULT_JSON:"):
            lines.append(line)
        else:
            lines.append(line + " " + "x" * (100 * 1024))
    evidence_path.write_text("\n".join(lines) + "\n")
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(app / "submission.json", submission)
    result = support._run_verifier(task, app, logs)
    assert result["evidence_validity"] == 1.0
    assert result["reward"] == pytest.approx(1.0)
