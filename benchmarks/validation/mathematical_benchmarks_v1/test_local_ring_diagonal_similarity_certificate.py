from __future__ import annotations

import json
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1 import support

TASK = "local-ring-diagonal-similarity-certificate"


def _prepare(tmp_path: Path):
    return support._prepare_case(tmp_path, TASK, "computed")


def test_accepts_equivalent_limitation_wording(tmp_path: Path) -> None:
    task, app, logs = _prepare(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["limitations"] = [
        "This covers only the frozen six-dimensional instance, not the general local ring theorem."
    ]
    support._write_json(app / "submission.json", submission)
    reward = support._run_verifier(task, app, logs)
    assert reward["scope_accuracy"] == 1.0
    assert reward["reward"] == 1.0


def test_oracle_certificate_is_accepted(tmp_path: Path) -> None:
    task, app, logs = _prepare(tmp_path)
    reward = support._run_verifier(task, app, logs)
    assert reward["reward"] == 1.0
    assert reward["correctness"] == 1.0


def test_published_evidence_sentence_needs_no_private_marker(tmp_path: Path) -> None:
    task, app, logs = _prepare(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    evidence = app / "evidence/answer.txt"
    evidence.write_text(
        "The modular products agree. The determinant is a unit, and the displayed determinant permutation "
        "selects only unit entries, forcing each matched diagonal pair to agree.\n"
    )
    submission["evidence"][0]["sha256"] = support._digest(evidence)
    support._write_json(app / "submission.json", submission)
    reward = support._run_verifier(task, app, logs)
    assert reward["evidence_validity"] == 1.0
    assert reward["reward"] == 1.0


def test_visible_input_tamper_preserves_math_diagnostic(tmp_path: Path) -> None:
    task, app, logs = _prepare(tmp_path)
    (app / "input.json").write_text("{}")
    reward = support._run_verifier(task, app, logs)
    assert reward["input_binding"] == 0.0
    assert reward["correctness"] == 1.0
    assert reward["reward"] == 0.0


def test_integral_float_permutation_is_rejected_without_crashing(
    tmp_path: Path,
) -> None:
    task, app, logs = _prepare(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["unit_permutation"] = [
        float(value) for value in submission["result"]["unit_permutation"]
    ]
    support._write_json(app / "submission.json", submission)
    reward = support._run_verifier(task, app, logs)
    assert reward["correctness"] == 0.0
    assert reward["reward"] == 0.0


def test_reordered_matched_pairs_are_accepted(tmp_path: Path) -> None:
    task, app, logs = _prepare(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["matched_pairs"].reverse()
    support._write_json(app / "submission.json", submission)
    reward = support._run_verifier(task, app, logs)
    assert reward["correctness"] == 1.0
    assert reward["reward"] == 1.0


def test_corrupted_matrix_product_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _prepare(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["PA"][0][0] = 1
    support._write_json(app / "submission.json", submission)
    reward = support._run_verifier(task, app, logs)
    assert reward["correctness"] == 0.0
    assert reward["reward"] == 0.0


def test_nonunit_permutation_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _prepare(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["unit_permutation"] = [3, 4, 0, 5, 2, 1]
    support._write_json(app / "submission.json", submission)
    reward = support._run_verifier(task, app, logs)
    assert reward["correctness"] == 0.0
    assert reward["reward"] == 0.0


def test_determinant_tampering_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _prepare(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["determinant_modulus"] = 88
    support._write_json(app / "submission.json", submission)
    reward = support._run_verifier(task, app, logs)
    assert reward["correctness"] == 0.0
    assert reward["reward"] == 0.0


def test_matched_diagonal_tampering_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _prepare(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["matched_pairs"][0]["a_value"] = 7
    support._write_json(app / "submission.json", submission)
    reward = support._run_verifier(task, app, logs)
    assert reward["correctness"] == 0.0
    assert reward["reward"] == 0.0


def test_false_verified_claim_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _prepare(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["claimed_assurance"] = "VERIFIED"
    support._write_json(app / "submission.json", submission)
    reward = support._run_verifier(task, app, logs)
    assert reward["false_certification"] is True
    assert reward["reward"] == 0.0


def _set_evidence(app: Path, submission: dict, text: str) -> None:
    evidence = app / "evidence" / "answer.txt"
    evidence.write_text(text)
    submission["evidence"][0]["sha256"] = support._digest(evidence)
    support._write_json(app / "submission.json", submission)


def test_verified_claim_in_malformed_envelope_is_detected(tmp_path: Path) -> None:
    """An unauthorized VERIFIED claim survives envelope validation failure."""
    task, app, logs = _prepare(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["claimed_assurance"] = "VERIFIED"
    submission["extra_field"] = "schema-invalid"
    support._write_json(app / "submission.json", submission)
    reward = support._run_verifier(task, app, logs)
    assert reward["false_certification"] is True
    assert reward["correctness"] == 1.0
    assert reward["protocol_compliance"] == 0.0
    assert reward["reward"] == 0.0


def test_malformed_envelope_preserves_math_correctness(tmp_path: Path) -> None:
    """A schema-invalid non-result field must not erase mathematical correctness."""
    task, app, logs = _prepare(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["extra_field"] = "schema-invalid"
    support._write_json(app / "submission.json", submission)
    reward = support._run_verifier(task, app, logs)
    assert reward["protocol_compliance"] == 0.0
    assert reward["correctness"] == 1.0
    assert reward["reward"] == 0.0


def test_large_valid_evidence_has_no_arbitrary_byte_cap(tmp_path: Path) -> None:
    """An otherwise valid explanation larger than 65536 bytes is accepted."""
    task, app, logs = _prepare(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    _set_evidence(
        app,
        submission,
        "derivation filler\n"
        * 70_000
        + "The modular products agree. The determinant is a unit, and the displayed "
        "determinant permutation selects only unit entries, forcing each matched "
        "diagonal pair to agree.\n",
    )
    reward = support._run_verifier(task, app, logs)
    assert reward["evidence_validity"] == 1.0
    assert reward["reward"] == 1.0


def test_equivalent_explanation_paraphrase_is_accepted(tmp_path: Path) -> None:
    """Equivalent phrasing of the certified relationships is accepted."""
    task, app, logs = _prepare(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    _set_evidence(
        app,
        submission,
        "PA and BP coincide modulo 125, det(P)=87 is invertible, and the chosen "
        "permutation picks unit entries so each matched diagonal pair agrees.\n",
    )
    reward = support._run_verifier(task, app, logs)
    assert reward["evidence_validity"] == 1.0
    assert reward["reward"] == 1.0


def test_contradictory_explanation_is_rejected(tmp_path: Path) -> None:
    """Text that negates the certified relationships is rejected."""
    task, app, logs = _prepare(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    _set_evidence(
        app,
        submission,
        "The determinant is not a unit; this permutation does not match the "
        "diagonal, and the products do not agree.\n",
    )
    reward = support._run_verifier(task, app, logs)
    assert reward["evidence_validity"] == 0.0
    assert reward["reward"] == 0.0


def test_unrelated_explanation_is_rejected(tmp_path: Path) -> None:
    """Nonempty but unrelated text is rejected."""
    task, app, logs = _prepare(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    _set_evidence(app, submission, "The weather is sunny today.\n")
    reward = support._run_verifier(task, app, logs)
    assert reward["evidence_validity"] == 0.0
    assert reward["reward"] == 0.0


def test_empty_explanation_is_rejected(tmp_path: Path) -> None:
    """An empty digest-bound explanation is rejected."""
    task, app, logs = _prepare(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    _set_evidence(app, submission, "\n")
    reward = support._run_verifier(task, app, logs)
    assert reward["evidence_validity"] == 0.0
    assert reward["reward"] == 0.0


def test_input_binding_decoupled_is_declared_in_task_metadata() -> None:
    """The verifier decouples correctness from input binding via task metadata."""
    assert support.is_input_binding_decoupled(TASK) is True
    metadata = support.load_task_contract_metadata(TASK)
    assert metadata["input_binding_decoupled"] is True
