from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import support

TASK = "newton-polygon-factorization-audit"


def _case(tmp_path: Path):
    return support._prepare_case(tmp_path, TASK, "computed")


def _rewrite(app: Path, submission: dict) -> None:
    _write_valid_newton_prose(app, submission)
    support._bind_result_evidence(app, submission)
    support._write_json(app / "submission.json", submission)


def test_accepts_equivalent_limitation_wording(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["limitations"] = [
        "The corrected Newton polygon lemma remains without formal verification."
    ]
    _rewrite(app, submission)
    accepted = support._run_verifier(task, app, logs)
    assert accepted["scope_accuracy"] == 1.0
    assert accepted["reward"] == 1.0


def test_accepts_an_alternative_prime_family_member(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"].update(
        {
            "prime": 3,
            "factor_left": ["9", "0", "1"],
            "factor_right": ["27", "0", "0", "0", "1"],
        }
    )
    _rewrite(app, submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


def test_plain_digest_bound_evidence_needs_no_private_marker(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    evidence = app / "evidence/answer.txt"
    evidence.write_text(
        "The Newton polygon lower hull has primitive negative edges. "
        "The old lemma hypotheses hold at the submitted indices, "
        "but the factor constant-term valuations contradict the old conclusion. "
        "The corrected left-edge primitivity condition fails (gcd > 1), "
        "so this witness does not refute the repair.\n"
    )
    submission["evidence"][0]["sha256"] = support._digest(evidence)
    support._write_json(app / "submission.json", submission)
    accepted = support._run_verifier(task, app, logs)
    assert accepted["evidence_validity"] == 1.0
    assert accepted["reward"] == 1.0


def test_accepts_numeric_explanation_without_oracle_phrases(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    evidence = app / "evidence/answer.txt"
    evidence.write_text(
        "The Newton polygon lower hull contains the submitted primitive edge, "
        "so the old right-edge hypotheses hold. The two factor constants have "
        "p-adic valuations 2 and 3; neither is zero. The corrected left-edge "
        "primitivity test fails because its gcd is greater than one.\n"
    )
    submission["evidence"][0]["sha256"] = support._digest(evidence)
    support._write_json(app / "submission.json", submission)
    accepted = support._run_verifier(task, app, logs)
    assert accepted["evidence_validity"] == 1.0
    assert accepted["reward"] == 1.0


def test_unverified_assurance_is_accepted(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["claimed_assurance"] = "UNVERIFIED"
    _rewrite(app, submission)
    accepted = support._run_verifier(task, app, logs)
    assert accepted["assurance_calibration"] == 1.0
    assert accepted["reward"] == 1.0


def test_rejects_witness_satisfying_corrected_left_conditions(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"].update(
        {
            "prime": 2,
            "factor_left": ["2", "0", "1"],
            "factor_right": ["2", "0", "0", "0", "1"],
            "ell": 2,
            "j": 6,
        }
    )
    _rewrite(app, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


@pytest.mark.parametrize(
    "corruption",
    ["composite", "tiny", "wrong_ell", "old_conclusion_true", "false_assurance"],
)
def test_rejects_invalid_or_nonrefuting_witnesses(
    tmp_path: Path, corruption: str
) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    result = submission["result"]
    if corruption == "composite":
        result["prime"] = 4
    elif corruption == "tiny":
        result.update(
            {
                "factor_left": ["2", "0", "1"],
                "factor_right": ["2", "0", "1"],
                "ell": 2,
                "j": 4,
            }
        )
    elif corruption == "wrong_ell":
        result["ell"] = 4
    elif corruption == "old_conclusion_true":
        result["factor_left"][0] = "1"
    else:
        submission["claimed_assurance"] = "VERIFIED"
    _rewrite(app, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["reward"] == 0.0


def test_rejects_visible_input_tampering(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    source = json.loads((app / "input.json").read_text())
    source["old_conclusion"] = "changed"
    support._write_json(app / "input.json", source)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["input_binding"] == 0.0
    assert rejected["correctness"] == 1.0
    assert rejected["reward"] == 0.0


def _write_valid_newton_prose(app: Path, submission: dict) -> None:
    """Write a valid Newton-polygon explanation and rebind the digest."""
    evidence = app / "evidence" / "answer.txt"
    evidence.write_text(
        "The Newton polygon lower hull has primitive negative edges. "
        "The old lemma hypotheses hold at the submitted indices, "
        "but the factor constant-term valuations contradict the old conclusion. "
        "The corrected left-edge primitivity condition fails (gcd > 1), "
        "so this witness does not refute the repair.\n"
    )
    submission["evidence"][0]["sha256"] = support._digest(evidence)


def test_empty_evidence_text_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    evidence = app / "evidence" / "answer.txt"
    evidence.write_text("")
    submission["evidence"][0]["sha256"] = support._digest(evidence)
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["evidence_validity"] == 0.0
    assert rejected["reward"] == 0.0


def test_unrelated_evidence_text_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
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


def test_evidence_missing_repair_concept_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    evidence = app / "evidence" / "answer.txt"
    evidence.write_text(
        "The Newton polygon valuation hull shows the old lemma "
        "conclusion is contradicted by the factor valuations.\n"
    )
    submission["evidence"][0]["sha256"] = support._digest(evidence)
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["evidence_validity"] == 0.0
    assert rejected["reward"] == 0.0


def test_horizontal_right_edge_is_rejected(tmp_path: Path) -> None:
    """When v(a_ell)==0 the edge is horizontal, not negative-slope."""
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    # p=2, factor_left=["4","0","1"], factor_right=["8","0","0","0","1","1"]
    # product = [32,0,8,0,4,8,2,1], degree 7
    # valuations: (0,5),(2,3),(4,2),(5,3),(6,1),(7,0)
    # ell=6, j=7: v_ell=1, v_j=0, gcd(1,1)=1 -> passes
    # But we need v_ell==0 case: use ell where v=0
    # product=[32,0,8,0,4,4,0,1]: v(0)=None at index 6, doesn't work
    # Let's construct: p=2, left=["4","0","1"], right=["8","0","0","0","1","2"]
    # product=[32,0,8,0,4,8,1,2], degree 7
    # vals: (0,5),(2,3),(4,2),(5,3),(6,0),(7,1)
    # ell=6, j=7: v_ell=0, v_j=1 -> v_j != 0, fails
    # Need v_j=0 and v_ell=0: impossible since gcd(0, delta) must be 1
    # and v_j=0 means the edge endpoint is on x-axis
    # With v_ell=0, edge is horizontal -> should be rejected
    # Construct: p=2, left=["2","0","1"], right=["8","0","0","0","1"]
    # product=[16,0,8,0,2,0,1], degree 6
    # vals: (0,4),(2,3),(4,1),(6,0)
    # ell=4, j=6: v_ell=1, v_j=0, gcd(1,2)=1 -> ok, negative slope
    # Need v_ell=0: try to get a point at (ell, 0) with ell < j and v_j=0
    # p=2, left=["4","0","1"], right=["8","0","0","0","1","1"]
    # product=[32,0,8,0,4,4,2,1], degree 7
    # vals: (0,5),(2,3),(4,2),(5,2),(6,1),(7,0)
    # ell=7 is j, not valid. ell=6, j=7: v_ell=1, v_j=0 -> ok
    # Try: p=2, left=["8","0","1"], right=["4","0","0","0","1","1"]
    # product=[32,0,8,0,4,4,1,1], degree 7
    # vals: (0,5),(2,3),(4,2),(5,2),(6,0),(7,0)
    # ell=6, j=7: v_ell=0, v_j=0, gcd(0,1)=1 -> currently passes!
    # Hull: (0,5)->(2,3)->(4,2)->(6,0)->(7,0)
    # Edge (6,0)->(7,0) is horizontal -> should be rejected
    submission["result"].update(
        {
            "prime": 2,
            "factor_left": ["8", "0", "1"],
            "factor_right": ["4", "0", "0", "0", "1", "1"],
            "ell": 6,
            "j": 7,
        }
    )
    _write_valid_newton_prose(app, submission)
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_accepts_valid_witness_with_j_less_than_degree(tmp_path: Path) -> None:
    """j need not equal the product degree; the edge at (ell,j) must be on the hull."""
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    # p=2, left=["4","0","1"], right=["8","0","0","0","1","2"]
    # product=[32,0,8,0,4,8,1,2], degree 7
    # vals: (0,5),(2,3),(4,2),(5,3),(6,0),(7,1)
    # hull: (0,5)->(2,3)->(6,0)->(7,1)
    # ell=2, j=6: edge (2,3)->(6,0) is on hull, primitive, negative slope
    # degree=7, j=6, so j < degree -> previously rejected by j!=degree
    # corrected left: v_zero=5, v_ell=3, diff=2, gcd(2,2)=2 -> fails (good)
    submission["result"].update(
        {
            "prime": 2,
            "factor_left": ["4", "0", "1"],
            "factor_right": ["8", "0", "0", "0", "1", "2"],
            "ell": 2,
            "j": 6,
        }
    )
    _write_valid_newton_prose(app, submission)
    support._write_json(app / "submission.json", submission)
    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


def test_malformed_envelope_preserves_correctness(tmp_path: Path) -> None:
    """An extra top-level field should not zero the correctness diagnostic."""
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["extra_field"] = "should fail protocol but not correctness"
    _write_valid_newton_prose(app, submission)
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["protocol_compliance"] == 0.0
    assert rejected["correctness"] == 1.0
    assert rejected["reward"] == 0.0


def test_index_beyond_product_degree_fails_closed(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["j"] = 24
    _write_valid_newton_prose(app, submission)
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_large_valid_evidence_is_accepted(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    evidence = app / "evidence" / "answer.txt"
    base = (
        "The Newton polygon lower hull has primitive negative edges. "
        "The old lemma hypotheses hold at the submitted indices, "
        "but the factor constant-term valuations contradict the old conclusion. "
        "The corrected left-edge primitivity condition fails (gcd > 1), "
        "so this witness does not refute the repair.\n"
    )
    evidence.write_text("derivation filler\n" * 70_000 + base)
    submission["evidence"][0]["sha256"] = support._digest(evidence)
    support._write_json(app / "submission.json", submission)
    accepted = support._run_verifier(task, app, logs)
    assert accepted["evidence_validity"] == 1.0
    assert accepted["reward"] == 1.0
