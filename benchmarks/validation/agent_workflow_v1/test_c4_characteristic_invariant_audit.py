from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.agent_workflow_v1 import support

TASK = "c4-characteristic-invariant-audit"


def test_oracle_passes(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    accepted = support._run_verifier(task, app, logs)
    assert accepted["reward"] == pytest.approx(1.0)


def test_rejects_induced_count_corruption(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["witnesses"][1]["induced_c4_count"] = 2
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0 and rejected["reward"] == 0.0


def test_rejects_chorded_graph_claimed_induced(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["witnesses"][2]["induced_c4_count"] = 1
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0 and rejected["reward"] == 0.0


def test_rejects_unsorted_edges(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["witnesses"][0]["edges"].reverse()
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0 and rejected["reward"] == 0.0


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
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


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
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_rejects_unhashable_role(tmp_path: Path) -> None:
    """A non-string role must not crash the set comprehension."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["witnesses"][0]["role"] = ["C4_FREE_ZERO_COUNT"]
    support._write_json(app / "submission.json", submission)
    result = support._run_verifier(task, app, logs)
    assert result["correctness"] == 0.0
    assert result["reward"] == 0.0


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
    assert result["evidence_validity"] == 0.0


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
    assert result["evidence_validity"] == 0.0


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
    assert rejected["reward"] == 0.0
    assert rejected["limitation_accuracy"] == 0.0


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
    assert rejected["reward"] == 0.0
    assert rejected["limitation_accuracy"] == 0.0


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
    assert rejected["reward"] == 0.0


def test_accepts_equivalent_no_claim_limitation(tmp_path: Path) -> None:
    """Equivalent negative limitation wording is part of the public contract."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["limitations"] = [
        "No Lean compilation or proof of the source-corrected conjecture is claimed."
    ]
    support._write_json(app / "submission.json", submission)
    accepted = support._run_verifier(task, app, logs)
    assert accepted["limitation_accuracy"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


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
    """Valid digest-bound evidence has no undocumented byte ceiling."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    evidence_path = app / "evidence" / "answer.txt"
    submission = json.loads((app / "submission.json").read_text())
    evidence_path.write_text(
        "induced characteristic not assessed\n"
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
