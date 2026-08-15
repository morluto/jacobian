from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import (
    _fixtures,
    _verifier,
)

TASK = "integer-perturbation-domain-audit"


def test_oracle_passes(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    accepted = _verifier._run_verifier(task, app, logs)
    assert accepted.reward == pytest.approx(1.0)


def test_accepts_alternative_periodic_witness(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
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
    _fixtures._bind_result_evidence(app, submission)
    _fixtures._write_json(app / "submission.json", submission)
    accepted = _verifier._run_verifier(task, app, logs)
    assert accepted.reward == pytest.approx(1.0)


def test_accepts_unordered_cancellation_indices(tmp_path: Path) -> None:
    """Cancellation indices are a set in the public contract, not a sequence."""
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["integer_witness"]["cancellation_indices"] = [2, 0]
    _fixtures._bind_result_evidence(app, submission)
    _fixtures._write_json(app / "submission.json", submission)
    accepted = _verifier._run_verifier(task, app, logs)
    assert accepted.details["correctness"] == 1.0
    assert accepted.reward == pytest.approx(1.0)


def test_accepts_witness_without_three_distinct_values(tmp_path: Path) -> None:
    """The visible contract does not require three distinct a or b values."""
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
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
    _fixtures._bind_result_evidence(app, submission)
    _fixtures._write_json(app / "submission.json", submission)
    accepted = _verifier._run_verifier(task, app, logs)
    assert accepted.details["correctness"] == 1.0
    assert accepted.reward == pytest.approx(1.0)


def test_rejects_missing_cancellation(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["integer_witness"]["cancellation_indices"] = [0]
    _fixtures._bind_result_evidence(app, submission)
    _fixtures._write_json(app / "submission.json", submission)
    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0 and rejected.reward == 0.0


def test_rejects_noninteger_domain_shortcut(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["integer_witness"]["b_values"][0] = 0
    _fixtures._bind_result_evidence(app, submission)
    _fixtures._write_json(app / "submission.json", submission)
    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0 and rejected.reward == 0.0


def test_rejects_boolean_lower_bound_certificates(tmp_path: Path) -> None:
    """JSON booleans must not satisfy integer lower-bound fields."""
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    redundancy = submission["result"]["nat_redundancy"]
    redundancy["a_lower_bound"] = False  # False == 0
    redundancy["b_lower_bound"] = True  # True == 1
    redundancy["sum_lower_bound"] = True  # True == 1
    _fixtures._bind_result_evidence(app, submission)
    _fixtures._write_json(app / "submission.json", submission)
    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_rejects_boolean_witness_extrema(tmp_path: Path) -> None:
    """JSON booleans must not satisfy b_min/b_max integer fields."""
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
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
    _fixtures._bind_result_evidence(app, submission)
    _fixtures._write_json(app / "submission.json", submission)
    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_rejects_symlinked_evidence_directory(tmp_path: Path) -> None:
    """A symlinked evidence/ directory must not escape the workspace."""
    import shutil

    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    forged_dir = app / "forged"
    forged_dir.mkdir()
    forged_answer = forged_dir / "answer.txt"
    submission = json.loads((app / "submission.json").read_text())
    original = (app / "evidence" / "answer.txt").read_text()
    forged_answer.write_text(original)
    submission["witness"][0]["sha256"] = _fixtures._digest(forged_answer)
    _fixtures._write_json(app / "submission.json", submission)
    shutil.rmtree(app / "evidence")
    (app / "evidence").symlink_to(forged_dir)
    result = _verifier._run_verifier(task, app, logs)
    assert result.reward == 0.0
    assert result.reward == 0.0


def test_deeply_nested_evidence_json_does_not_crash(tmp_path: Path) -> None:
    """A deeply nested RESULT_JSON line must not crash the verifier."""
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    evidence_path = app / "evidence" / "answer.txt"
    nested = "null" + ",[" * 200 + "]" * 200
    evidence_path.write_text(f"RESULT_JSON: {nested}\nnatural integer not assessed\n")
    submission = json.loads((app / "submission.json").read_text())
    submission["witness"][0]["sha256"] = _fixtures._digest(evidence_path)
    _fixtures._write_json(app / "submission.json", submission)
    result = _verifier._run_verifier(task, app, logs)
    assert result.reward == 0.0
    assert result.reward == 0.0


def test_rejects_assertion_style_theorem_overclaim(tmp_path: Path) -> None:
    """A theorem can be overclaimed without using proof or verification verbs."""
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
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
    submission["witness"][0]["sha256"] = _fixtures._digest(evidence_path)
    _fixtures._write_json(app / "submission.json", submission)
    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.reward == 0.0
    assert rejected.reward == 0.0


def test_accepts_equivalent_concise_audit_evidence(tmp_path: Path) -> None:
    """Equivalent Nat/Z wording satisfies the public concise-audit requirement."""
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
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
    submission["witness"][0]["sha256"] = _fixtures._digest(evidence_path)
    _fixtures._write_json(app / "submission.json", submission)
    accepted = _verifier._run_verifier(task, app, logs)
    assert accepted.reward == 1.0
    assert accepted.reward == pytest.approx(1.0)


def test_rejects_multiple_result_markers(tmp_path: Path) -> None:
    """Evidence must bind exactly one unambiguous result marker."""
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    evidence_path = app / "evidence" / "answer.txt"
    original = evidence_path.read_text()
    marker = next(
        line for line in original.splitlines() if line.startswith("RESULT_JSON:")
    )
    evidence_path.write_text(original + marker + "\n")
    submission = json.loads((app / "submission.json").read_text())
    submission["witness"][0]["sha256"] = _fixtures._digest(evidence_path)
    _fixtures._write_json(app / "submission.json", submission)
    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.reward == 0.0
    assert rejected.reward == 0.0


def test_rejects_affirmative_irrationality_claim_in_evidence(tmp_path: Path) -> None:
    """Evidence prose must obey the irrationality limitation."""
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    evidence_path = app / "evidence" / "answer.txt"
    evidence_path.write_text(
        "natural integer perturbations are not assessed. "
        "The source irrationality theorem has been proved.\n"
        + evidence_path.read_text()
    )
    submission = json.loads((app / "submission.json").read_text())
    submission["witness"][0]["sha256"] = _fixtures._digest(evidence_path)
    _fixtures._write_json(app / "submission.json", submission)
    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.reward == 0.0


def test_rejects_bool_int_evidence_marker_coercion(tmp_path: Path) -> None:
    """Digest-bound JSON must preserve exact integer-versus-boolean types."""
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    evidence_path = app / "evidence" / "answer.txt"
    submission = json.loads((app / "submission.json").read_text())
    marker_result = json.loads(json.dumps(submission["result"]))
    marker_result["nat_redundancy"]["a_lower_bound"] = False
    evidence_path.write_text(
        "natural integer not assessed\nRESULT_JSON: "
        + json.dumps(marker_result, sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    submission["witness"][0]["sha256"] = _fixtures._digest(evidence_path)
    _fixtures._write_json(app / "submission.json", submission)
    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.reward == 0.0


def test_input_tamper_preserves_math_correctness(tmp_path: Path) -> None:
    """A tampered workspace input must not zero mathematical correctness."""
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    (app / "input.json").write_text("{}")
    result = _verifier._run_verifier(task, app, logs)
    assert result.details["correctness"] == 1.0
    assert result.details["input_binding"] == 0.0
    assert result.reward == 0.0


def test_oversized_workspace_input_fails_closed(tmp_path: Path) -> None:
    """An oversized workspace input must fail closed without crashing."""
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    (app / "input.json").write_text("x" * (20 * 1024 * 1024))
    result = _verifier._run_verifier(task, app, logs)
    assert result.details["input_binding"] == 0.0
    assert result.reward == 0.0


def test_large_valid_evidence_is_accepted(tmp_path: Path) -> None:
    """No undocumented evidence size cap; large valid evidence passes."""
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
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
    submission["witness"][0]["sha256"] = _fixtures._digest(evidence_path)
    _fixtures._write_json(app / "submission.json", submission)
    result = _verifier._run_verifier(task, app, logs)
    assert result.reward == 1.0
    assert result.reward == pytest.approx(1.0)
