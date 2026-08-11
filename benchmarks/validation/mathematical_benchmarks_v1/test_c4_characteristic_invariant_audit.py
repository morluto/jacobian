from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import support

TASK = "c4-characteristic-invariant-audit"


def test_oracle_passes(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    accepted = support._run_verifier(task, app, logs)
    assert accepted.reward == pytest.approx(1.0)


def test_rejects_induced_count_corruption(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["witnesses"][1]["induced_c4_count"] = 2
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0 and rejected.reward == 0.0


def test_rejects_chorded_graph_claimed_induced(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["witnesses"][2]["induced_c4_count"] = 1
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0 and rejected.reward == 0.0


def test_rejects_unsorted_edges(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["witnesses"][0]["edges"].reverse()
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0 and rejected.reward == 0.0


def test_rejects_oversized_edge_list_before_graph_checks(tmp_path: Path) -> None:
    """The public edge bound must be enforced before pair materialization."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["witnesses"][0]["edges"] = [[0, 1]] * 37
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_rejects_above_ceiling_assurance(tmp_path: Path) -> None:
    """CHECKED is above the COMPUTED ceiling and must fail closed."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["claimed_assurance"] = "CHECKED"
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.reward == 0.0
    assert rejected.details["assurance_calibration"] == 0.0
    assert rejected.details["evidence_validity"] == 1.0


def test_rejects_boolean_invariant_counts(tmp_path: Path) -> None:
    """JSON booleans must not satisfy integer invariant fields."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    # C4_FREE witness has induced_c4_count=0, c4_free_characteristic=1
    submission["result"]["witnesses"][0]["induced_c4_count"] = False  # False == 0
    submission["result"]["witnesses"][0]["c4_free_characteristic"] = True  # True == 1
    support._bind_result_evidence(app, submission)
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_rejects_boolean_chorded_zero_fields(tmp_path: Path) -> None:
    """Boolean zero fields on the chorded witness must not pass."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    # CHORDED witness has induced_c4_count=0, c4_free_characteristic=0
    submission["result"]["witnesses"][2]["induced_c4_count"] = False
    submission["result"]["witnesses"][2]["c4_free_characteristic"] = False
    support._bind_result_evidence(app, submission)
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_rejects_unhashable_role(tmp_path: Path) -> None:
    """A non-string role must not crash the set comprehension."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["witnesses"][0]["role"] = ["C4_FREE_ZERO_COUNT"]
    support._write_json(app / "submission.json", submission)
    result = support._run_verifier(task, app, logs)
    assert result.details["correctness"] == 0.0
    assert result.reward == 0.0


def test_rejects_symlinked_evidence_directory(tmp_path: Path) -> None:
    """A symlinked evidence/ directory must not escape the workspace."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    # Replace evidence/ with a symlink to a temp dir containing a forged answer.txt
    import shutil

    forged_dir = app / "forged"
    forged_dir.mkdir()
    forged_answer = forged_dir / "answer.txt"
    submission = json.loads((app / "submission.json").read_text())
    # Write a valid-looking evidence file in the forged directory
    original = (app / "evidence" / "answer.txt").read_text()
    forged_answer.write_text(original)
    submission["evidence"][0]["sha256"] = support._digest(forged_answer)
    support._write_json(app / "submission.json", submission)
    # Replace evidence/ with symlink
    shutil.rmtree(app / "evidence")
    (app / "evidence").symlink_to(forged_dir)
    result = support._run_verifier(task, app, logs)
    assert result.details["evidence_validity"] == 0.0


def test_deeply_nested_evidence_json_does_not_crash(tmp_path: Path) -> None:
    """A deeply nested RESULT_JSON line must not crash the verifier."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    evidence_path = app / "evidence" / "answer.txt"
    nested = "null" + ",[" * 200 + "]" * 200
    evidence_path.write_text(
        f"RESULT_JSON: {nested}\ninduced characteristic not assessed\n"
    )
    submission = json.loads((app / "submission.json").read_text())
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(app / "submission.json", submission)
    result = support._run_verifier(task, app, logs)
    assert result.details["evidence_validity"] == 0.0


def test_rejects_affirmative_lean_verification_claim(tmp_path: Path) -> None:
    """Limitations that affirm Lean verification must be rejected."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["limitations"] = [
        "Lean compilation is not assessed.",
        "The Lean theorem has been verified and compiles correctly.",
    ]
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.reward == 0.0
    assert rejected.details["limitation_accuracy"] == 0.0


def test_rejects_lean_compile_overclaim(tmp_path: Path) -> None:
    """Compilation claims are forbidden even without verified/proved."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["limitations"] = [
        "Lean compilation is not assessed.",
        "The upstream Lean theorem compiles correctly.",
    ]
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.reward == 0.0
    assert rejected.details["limitation_accuracy"] == 0.0


def test_rejects_lean_compile_overclaim_across_limitation_items(
    tmp_path: Path,
) -> None:
    """A compile claim in a later limitation item must remain prohibited."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["limitations"] = [
        "Lean compilation is not assessed.",
        "The upstream theorem compiles correctly.",
        "The source-corrected conjecture is not claimed.",
    ]
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["limitation_accuracy"] == 0.0


def test_rejects_lean_compile_overclaim_across_evidence_lines(
    tmp_path: Path,
) -> None:
    """Evidence lines must not hide a compile claim behind a disclaimer."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    evidence_path = app / "evidence" / "answer.txt"
    submission = json.loads((app / "submission.json").read_text())
    evidence_path.write_text(
        "Lean compilation is not assessed.\n"
        "The upstream theorem compiles correctly.\n"
        "The induced count and characteristic are audited; no proof of the "
        "source-corrected conjecture is claimed.\n"
        "RESULT_JSON: "
        + json.dumps(submission["result"], sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["evidence_validity"] == 0.0


def test_accepts_finite_verification_wording_with_limitations(
    tmp_path: Path,
) -> None:
    """Finite certificate verification is not an upstream Lean overclaim."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["limitations"] = [
        "The finite graph certificates were verified by exhaustive enumeration; "
        "Lean compilation is not assessed and no proof of the source-corrected "
        "conjecture is claimed."
    ]
    support._write_json(app / "submission.json", submission)
    accepted = support._run_verifier(task, app, logs)
    assert accepted.details["limitation_accuracy"] == 1.0
    assert accepted.reward == pytest.approx(1.0)


def test_requires_topic_specific_limitation_negation(tmp_path: Path) -> None:
    """Lean negation must not satisfy the conjecture limitation."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["limitations"] = [
        "Lean compilation is not assessed for this finite audit; the "
        "source-corrected conjecture remains the target."
    ]
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["limitation_accuracy"] == 0.0
    assert rejected.reward == 0.0


def test_rejects_lean_compile_overclaim_in_evidence(tmp_path: Path) -> None:
    """Evidence prose must obey the same prohibition as limitations."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    evidence_path = app / "evidence" / "answer.txt"
    submission = json.loads((app / "submission.json").read_text())
    evidence_path.write_text(
        "induced characteristic not assessed. The upstream Lean theorem compiles correctly.\n"
        + evidence_path.read_text()
    )
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["evidence_validity"] == 0.0


def test_rejects_split_lean_compile_overclaim(tmp_path: Path) -> None:
    """A topic in one clause must not hide a claim in the next clause."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["limitations"] = [
        "Lean compilation is not assessed and the theorem compiles correctly.",
        "The source-corrected conjecture is not claimed.",
    ]
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["limitation_accuracy"] == 0.0


def test_rejects_elided_subject_compile_overclaim(tmp_path: Path) -> None:
    """A coordinated predicate inherits its upstream theorem subject."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["limitations"] = [
        "The upstream Lean theorem is not assessed and compiles correctly; "
        "no proof of the source-corrected conjecture is claimed."
    ]
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["limitation_accuracy"] == 0.0
    assert rejected.reward == 0.0


@pytest.mark.parametrize("intervening", ["no proof is claimed", "is not verified"])
def test_retains_theorem_context_across_negated_predicates(
    tmp_path: Path, intervening: str
) -> None:
    """Negated coordinated predicates must not clear the inherited subject."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["limitations"] = [
        f"The upstream Lean theorem is not assessed and {intervening} but "
        "compiles correctly; no proof of the source-corrected conjecture is claimed."
    ]
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["limitation_accuracy"] == 0.0
    assert rejected.reward == 0.0


@pytest.mark.parametrize("predicate", ["can compile", "does compile", "has compiled"])
def test_rejects_modal_elided_subject_compile_overclaim(
    tmp_path: Path, predicate: str
) -> None:
    """Modal and auxiliary predicates inherit the upstream theorem subject."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["limitations"] = [
        f"The upstream Lean theorem is not assessed and {predicate} correctly; "
        "no proof of the source-corrected conjecture is claimed."
    ]
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["limitation_accuracy"] == 0.0
    assert rejected.reward == 0.0


def test_rejects_elided_subject_compile_overclaim_in_evidence(
    tmp_path: Path,
) -> None:
    """Evidence must retain theorem context across a conjunction too."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    evidence_path = app / "evidence" / "answer.txt"
    submission = json.loads((app / "submission.json").read_text())
    evidence_path.write_text(
        "The upstream Lean theorem is not assessed and compiles correctly; "
        "no proof of the source-corrected conjecture is claimed.\n"
        "The induced count and characteristic are audited.\n"
        "RESULT_JSON: "
        + json.dumps(submission["result"], sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["evidence_validity"] == 0.0
    assert rejected.reward == 0.0


def test_accepts_explicit_finite_subject_after_theorem_limitation(
    tmp_path: Path,
) -> None:
    """An explicit finite-audit subject must not inherit theorem context."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["limitations"] = [
        "The upstream Lean theorem is not assessed and the finite graph "
        "certificates were verified by exhaustive enumeration; no proof of "
        "the source-corrected conjecture is claimed."
    ]
    support._write_json(app / "submission.json", submission)
    accepted = support._run_verifier(task, app, logs)
    assert accepted.details["limitation_accuracy"] == 1.0
    assert accepted.reward == pytest.approx(1.0)


def test_accepts_theorem_object_in_finite_certificate_claim(tmp_path: Path) -> None:
    """A theorem object must not replace an explicit finite-certificate subject."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["limitations"] = [
        "Lean compilation is not assessed.",
        "The finite graph certificates were verified against the upstream Lean "
        "theorem and compile correctly.",
        "No proof of the source-corrected conjecture is claimed.",
    ]
    support._write_json(app / "submission.json", submission)
    accepted = support._run_verifier(task, app, logs)
    assert accepted.details["limitation_accuracy"] == 1.0
    assert accepted.reward == pytest.approx(1.0)


def test_accepts_equivalent_evidence_limitation_wording(tmp_path: Path) -> None:
    """Evidence may use an equivalent no-claim limitation without a keyword."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    evidence_path = app / "evidence" / "answer.txt"
    submission = json.loads((app / "submission.json").read_text())
    evidence_path.write_text(
        "The induced count and characteristic are audited. No Lean compilation "
        "or proof of the source-corrected conjecture is claimed.\n"
        + "RESULT_JSON: "
        + json.dumps(submission["result"], sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(app / "submission.json", submission)
    accepted = support._run_verifier(task, app, logs)
    assert accepted.details["evidence_validity"] == 1.0


def test_assurance_diagnostic_is_independent_of_envelope(tmp_path: Path) -> None:
    """An envelope error must not erase a separately correct assurance."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["conclusion"] = "UNSUPPORTED"
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["assurance_calibration"] == 1.0
    assert rejected.reward == 0.0


def test_rejects_missing_conjecture_limitation(tmp_path: Path) -> None:
    """Both the Lean and corrected-conjecture limitations are required."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["limitations"] = ["No Lean compilation is claimed."]
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["limitation_accuracy"] == 0.0


def test_rejects_affirmative_conjecture_proof_claim(tmp_path: Path) -> None:
    """Limitations that affirm the corrected conjecture is proved must be rejected."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["limitations"] = [
        "Lean compilation is not assessed.",
        "The source-corrected conjecture has been proved.",
    ]
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.reward == 0.0


def test_rejects_noun_form_conjecture_proof_claim(tmp_path: Path) -> None:
    """Noun-form proof assertions remain prohibited overclaims."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["limitations"] = [
        "Lean compilation is not assessed.",
        "No proof of the source-corrected conjecture is claimed; the "
        "source-corrected conjecture has a proof.",
    ]
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["limitation_accuracy"] == 0.0
    assert rejected.reward == 0.0


def test_accepts_equivalent_no_claim_limitation(tmp_path: Path) -> None:
    """Equivalent negative limitation wording is part of the public contract."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["limitations"] = [
        "No Lean compilation or proof of the source-corrected conjecture is claimed."
    ]
    support._write_json(app / "submission.json", submission)
    accepted = support._run_verifier(task, app, logs)
    assert accepted.details["limitation_accuracy"] == 1.0
    assert accepted.reward == pytest.approx(1.0)


def test_accepts_result_marker_after_prose(tmp_path: Path) -> None:
    """Evidence prose may appear after the result marker."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    evidence_path = app / "evidence" / "answer.txt"
    submission = json.loads((app / "submission.json").read_text())
    evidence_path.write_text(
        "RESULT_JSON: "
        + json.dumps(submission["result"], sort_keys=True, separators=(",", ":"))
        + "\nThe induced count and characteristic are not assessed; no Lean "
        "compilation or proof of the corrected conjecture is claimed.\n"
    )
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(app / "submission.json", submission)
    accepted = support._run_verifier(task, app, logs)
    assert accepted.details["evidence_validity"] == 1.0


def test_rejects_bool_int_coercion_in_result_marker(tmp_path: Path) -> None:
    """Digest-bound JSON must preserve exact integer-versus-boolean types."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    evidence_path = app / "evidence" / "answer.txt"
    submission = json.loads((app / "submission.json").read_text())
    marker_result = json.loads(json.dumps(submission["result"]))
    marker_result["witnesses"][0]["induced_c4_count"] = False
    evidence_path.write_text(
        "induced characteristic not assessed\nRESULT_JSON: "
        + json.dumps(marker_result, sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["evidence_validity"] == 0.0


def test_scope_diagnostic_is_independent_of_contract(tmp_path: Path) -> None:
    """An unsupported assurance must not erase a correct scope diagnostic."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["claimed_assurance"] = "CHECKED"
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["scope_accuracy"] == 1.0
    assert rejected.details["assurance_calibration"] == 0.0


def test_incomplete_scope_gates_reward(tmp_path: Path) -> None:
    """An independently reported scope failure must still fail closed."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["scope"] = "partial finite comparison"
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 1.0
    assert rejected.details["scope_accuracy"] == 0.0
    assert rejected.reward == 0.0


def test_completeness_diagnostic_is_independent(tmp_path: Path) -> None:
    """Completeness failures must be visible without erasing other dimensions."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["completeness"] = "PARTIAL"
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 1.0
    assert rejected.details["scope_accuracy"] == 1.0
    assert rejected.details["assurance_calibration"] == 1.0
    assert rejected.details["completeness_accuracy"] == 0.0
    assert rejected.reward == 0.0


def test_input_tamper_preserves_math_correctness(tmp_path: Path) -> None:
    """A tampered workspace input must not zero mathematical correctness."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    (app / "input.json").write_text("{}")
    result = support._run_verifier(task, app, logs)
    assert result.details["correctness"] == 1.0
    assert result.details["input_binding"] == 0.0
    assert result.reward == 0.0


def test_oversized_workspace_input_fails_closed(tmp_path: Path) -> None:
    """An oversized workspace input must fail closed without crashing."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    (app / "input.json").write_text("x" * (20 * 1024 * 1024))
    result = support._run_verifier(task, app, logs)
    assert result.details["input_binding"] == 0.0
    assert result.reward == 0.0


def test_accepts_large_digest_bound_evidence(tmp_path: Path) -> None:
    """Valid digest-bound evidence has no undocumented byte ceiling."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    evidence_path = app / "evidence" / "answer.txt"
    submission = json.loads((app / "submission.json").read_text())
    evidence_path.write_text(
        "The induced count and characteristic are not assessed; no Lean "
        "compilation or proof of the corrected conjecture is claimed.\n"
        + ("audit\n" * 200_000)
        + "RESULT_JSON: "
        + json.dumps(submission["result"], sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(app / "submission.json", submission)
    accepted = support._run_verifier(task, app, logs)
    assert accepted.details["evidence_validity"] == 1.0
    assert accepted.reward == pytest.approx(1.0)


def test_rejects_oversized_evidence_line_after_valid_content(tmp_path: Path) -> None:
    """A hostile trailing line must not bypass the bounded evidence scan."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    evidence_path = app / "evidence" / "answer.txt"
    submission = json.loads((app / "submission.json").read_text())
    evidence_path.write_text(
        "The induced count and characteristic are not assessed; no Lean "
        "compilation or proof of the corrected conjecture is claimed.\n"
        + "RESULT_JSON: "
        + json.dumps(submission["result"], sort_keys=True, separators=(",", ":"))
        + "\n"
        + ("x" * (1024 * 1024 + 1))
        + "\n"
    )
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["evidence_validity"] == 0.0
    assert rejected.reward == 0.0
