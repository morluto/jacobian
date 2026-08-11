from __future__ import annotations

import json
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1 import support

TASK = "finite-field-irreducibility-repair"


def _case(tmp_path: Path):
    return support._prepare_case(tmp_path, TASK, "computed")


def test_reference_passes(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    assert support._run_verifier(task, app, logs).reward == 1.0


def _reject(tmp_path: Path, mutate) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    mutate(submission)
    support._bind_result_evidence(app, submission)
    support._write_json(path, submission)
    assert support._run_verifier(task, app, logs).reward == 0.0


def test_wrong_bad_factor_is_rejected(tmp_path: Path) -> None:
    _reject(tmp_path, lambda s: s["result"].__setitem__("bad_factor", [0, 1]))


def test_corrupt_p2_remainder_is_rejected(tmp_path: Path) -> None:
    _reject(tmp_path, lambda s: s["result"]["p2_remainder"].__setitem__(0, 0))


def test_reducible_repair_prime_is_rejected(tmp_path: Path) -> None:
    _reject(tmp_path, lambda s: s["result"].__setitem__("repair_prime", 3))


def test_false_verified_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["claimed_assurance"] = "VERIFIED"
    support._write_json(path, submission)
    result = support._run_verifier(task, app, logs)
    assert result.reward == 0.0
    assert result.details["false_certification"] is True


def test_false_verified_keeps_other_diagnostics_independent(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["claimed_assurance"] = "VERIFIED"
    support._write_json(path, submission)
    result = support._run_verifier(task, app, logs)
    assert result.reward == 0.0
    assert result.details["correctness"] == 1.0
    assert result.details["evidence_validity"] == 1.0
    assert result.details["scope_accuracy"] == 1.0
    assert result.details["assurance_calibration"] == 0.0


def test_boolean_fixed_coefficients_are_rejected(tmp_path: Path) -> None:
    _reject(
        tmp_path,
        lambda s: s["result"].update(
            bad_reduction=[True, False, False, False, True],
            bad_factor=[True, True],
            rabin_gcd_degree=False,
        ),
    )


def test_unrelated_explanation_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    evidence = app / "evidence" / "answer.txt"
    marker = next(
        line
        for line in evidence.read_text().splitlines()
        if line.startswith("RESULT_JSON:")
    )
    evidence.write_text(
        ("reducible rabin gcd rational density " * 30) + "\n" + marker + "\n"
    )
    submission = json.loads((app / "submission.json").read_text())
    submission["evidence"][0]["sha256"] = support._digest(evidence)
    support._write_json(app / "submission.json", submission)
    assert support._run_verifier(task, app, logs).reward == 0.0


def test_density_limitation_is_exact(tmp_path: Path) -> None:
    _reject(
        tmp_path,
        lambda s: s.__setitem__("limitations", ["The density is not purple."]),
    )


def test_visible_input_tampering_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    data = json.loads((app / "input.json").read_text())
    data["source"]["row"] = 3
    support._write_json(app / "input.json", data)
    assert support._run_verifier(task, app, logs).reward == 0.0


def _rewrite_prose(app: Path, prose: str) -> None:
    """Replace the evidence prose while preserving the RESULT_JSON marker."""

    evidence = app / "evidence" / "answer.txt"
    marker = next(
        line
        for line in evidence.read_text().splitlines()
        if line.startswith("RESULT_JSON:")
    )
    evidence.write_text(prose + "\n" + marker + "\n")
    submission = json.loads((app / "submission.json").read_text())
    submission["evidence"][0]["sha256"] = support._digest(evidence)
    support._write_json(app / "submission.json", submission)


_REFERENCE_PROSE = (
    "The reduction modulo two is reducible, since x^4+1=(x+1)^4 there, "
    "so it cannot support the published irreducibility implication. "
    "Modulo eleven, Rabin's degree-four test succeeds: the p^4 remainder is x "
    "and the gcd of f with x^(p^2)-x is one. "
    "Hence the primitive polynomial is irreducible over the rationals. "
    "This repairs only the irreducibility step and does not verify the "
    "later Galois-group or density claims."
)


def test_truthful_proof_assistant_disclaimer_is_accepted(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    _rewrite_prose(
        app, _REFERENCE_PROSE + " This result was not checked by a proof assistant."
    )
    result = support._run_verifier(task, app, logs)
    assert result.details["evidence_validity"] == 1.0
    assert result.reward == 1.0


def test_asserted_proof_assistant_verification_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    _rewrite_prose(
        app, _REFERENCE_PROSE + " This result was verified by a proof assistant."
    )
    result = support._run_verifier(task, app, logs)
    assert result.details["evidence_validity"] == 0.0
    assert result.reward == 0.0


def test_evidence_validity_independent_of_correctness(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["result"]["bad_factor"] = [0, 1]
    support._bind_result_evidence(app, submission)
    support._write_json(path, submission)
    result = support._run_verifier(task, app, logs)
    assert result.details["correctness"] == 0.0
    assert result.details["evidence_validity"] == 1.0
    assert result.reward == 0.0


def test_synonymous_formal_verification_claim_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    _rewrite_prose(app, _REFERENCE_PROSE + " Lean 4 formally verified this proof.")
    result = support._run_verifier(task, app, logs)
    assert result.details["evidence_validity"] == 0.0
    assert result.reward == 0.0


def test_synonymous_machine_checked_claim_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    _rewrite_prose(app, _REFERENCE_PROSE + " This result was machine-checked by Coq.")
    result = support._run_verifier(task, app, logs)
    assert result.details["evidence_validity"] == 0.0
    assert result.reward == 0.0


def test_truthful_formal_verification_disclaimer_is_accepted(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    _rewrite_prose(
        app,
        _REFERENCE_PROSE
        + " This result was not formally verified by any theorem prover.",
    )
    result = support._run_verifier(task, app, logs)
    assert result.details["evidence_validity"] == 1.0
    assert result.reward == 1.0


def _rewrite_marker(app: Path, marker_result: dict) -> None:
    """Replace the RESULT_JSON marker while preserving the prose."""

    evidence = app / "evidence" / "answer.txt"
    lines = evidence.read_text().splitlines()
    prose_lines = [line for line in lines if not line.startswith("RESULT_JSON:")]
    marker = "RESULT_JSON:" + json.dumps(
        marker_result, sort_keys=True, separators=(",", ":")
    )
    evidence.write_text("\n".join(prose_lines) + "\n" + marker + "\n")
    submission = json.loads((app / "submission.json").read_text())
    submission["evidence"][0]["sha256"] = support._digest(evidence)
    support._write_json(app / "submission.json", submission)


def test_boolean_in_marker_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    marker_result = dict(submission["result"])
    marker_result["rabin_gcd_degree"] = False
    _rewrite_marker(app, marker_result)
    result = support._run_verifier(task, app, logs)
    assert result.details["evidence_validity"] == 0.0
    assert result.reward == 0.0


def test_boolean_coefficient_in_marker_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    marker_result = dict(submission["result"])
    marker_result["bad_reduction"] = [True, False, False, False, True]
    _rewrite_marker(app, marker_result)
    result = support._run_verifier(task, app, logs)
    assert result.details["evidence_validity"] == 0.0
    assert result.reward == 0.0
