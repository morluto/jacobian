from __future__ import annotations

import json
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1 import support

TASK = "fractional-ratio-proof-repair"


def _run(tmp_path: Path, mutate=None):
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    if mutate:
        mutate(submission)
        support._bind_result_evidence(app, submission)
        support._write_json(app / "submission.json", submission)
    return support._run_verifier(task, app, logs)


def test_accepts_equivalent_limitation_wording(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        lambda submission: submission.update(
            limitations=[
                "This exact 24-item binary instance does not prove a general greedy theorem."
            ]
        ),
    )
    assert result["scope_accuracy"] == 1.0
    assert result["reward"] == 1.0


def test_oracle_passes(tmp_path: Path) -> None:
    assert _run(tmp_path)["reward"] == 1.0


def test_published_evidence_sentence_needs_no_private_marker(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    evidence = app / "evidence/answer.txt"
    evidence.write_text(
        "The public proof replaces the ratio objective, relaxes the binary domain, and adds an undeclared budget. "
        "The exact residual certificate repairs the frozen objective: every coordinate is chosen by its signed "
        "residual and the maximum transformed residual is zero.\n"
    )
    submission["evidence"][0]["sha256"] = support._digest(evidence)
    support._write_json(app / "submission.json", submission)
    result = support._run_verifier(task, app, logs)
    assert result["evidence_validity"] == 1.0
    assert result["reward"] == 1.0


def test_visible_input_tamper_preserves_math_diagnostic(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    (app / "input.json").write_text("{}")
    result = support._run_verifier(task, app, logs)
    assert result["input_binding"] == 0.0
    assert result["correctness"] == 1.0
    assert result["reward"] == 0.0


def test_reordered_indexed_certificate_is_accepted(tmp_path: Path) -> None:
    def mutate(s):
        s["result"]["selected_indices"].reverse()
        s["result"]["item_residuals"].reverse()
        s["result"]["positive_residual_indices"].reverse()

    assert _run(tmp_path, mutate)["reward"] == 1.0


def test_oversized_ratio_is_rejected_without_crashing(tmp_path: Path) -> None:
    def mutate(s):
        s["result"]["attained_ratio"] = "1" * 5000 + "/1"

    result = _run(tmp_path, mutate)
    assert result["correctness"] == 0.0
    assert result["reward"] == 0.0


def test_rejects_omitted_contract_mismatch(tmp_path: Path) -> None:
    def mutate(s):
        s["result"]["contract_mismatches"].pop()

    assert _run(tmp_path, mutate)["reward"] == 0.0


def test_rejects_suboptimal_selection(tmp_path: Path) -> None:
    def mutate(s):
        s["result"]["selected_indices"] = [0, 2]

    assert _run(tmp_path, mutate)["reward"] == 0.0


def test_rejects_corrupted_item_residual(tmp_path: Path) -> None:
    def mutate(s):
        s["result"]["item_residuals"][7]["value"] += 1

    assert _run(tmp_path, mutate)["reward"] == 0.0


def test_boolean_constant_residual_is_rejected(tmp_path: Path) -> None:
    result = _run(
        tmp_path, lambda s: s["result"].__setitem__("constant_residual", False)
    )
    assert result["correctness"] == 0.0
    assert result["reward"] == 0.0


def test_bad_scope_preserves_math_diagnostic(tmp_path: Path) -> None:
    result = _run(tmp_path, lambda s: s.__setitem__("scope", "wrong"))
    assert result["correctness"] == 1.0
    assert result["scope_accuracy"] == 0.0
    assert result["reward"] == 0.0


def test_rejects_corrupted_ratio(tmp_path: Path) -> None:
    def mutate(s):
        s["result"]["attained_ratio"] = "2/1"

    assert _run(tmp_path, mutate)["reward"] == 0.0


def test_rejects_false_verified_claim(tmp_path: Path) -> None:
    def mutate(s):
        s["claimed_assurance"] = "VERIFIED"

    result = _run(tmp_path, mutate)
    assert result["false_certification"] is True
    assert result["reward"] == 0.0


def test_unhashable_mismatch_entry_does_not_crash(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["contract_mismatches"] = [{}]
    support._bind_result_evidence(app, submission)
    support._write_json(app / "submission.json", submission)
    result = support._run_verifier(task, app, logs)
    assert result["correctness"] == 0.0
    assert result["reward"] == 0.0


def test_float_maximum_residual_sum_is_rejected(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        lambda s: s["result"].__setitem__("maximum_residual_sum", 0.0),
    )
    assert result["correctness"] == 0.0
    assert result["reward"] == 0.0


def test_equivalent_evidence_phrasing_is_accepted(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    evidence = app / "evidence/answer.txt"
    evidence.write_text(
        "The published analysis substitutes a linear benefit for the true ratio "
        "objective. It permits fractional values instead of the binary domain "
        "constraint. A hidden budget constraint is introduced that is absent from "
        "the frozen problem. The exact residual certificate restores the original "
        "objective: each coordinate is selected according to its signed residual, "
        "and the maximum transformed residual sum equals zero.\n"
    )
    submission["evidence"][0]["sha256"] = support._digest(evidence)
    support._write_json(app / "submission.json", submission)
    result = support._run_verifier(task, app, logs)
    assert result["evidence_validity"] == 1.0
    assert result["reward"] == 1.0


def test_unrelated_evidence_text_is_rejected(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    evidence = app / "evidence/answer.txt"
    evidence.write_text(
        "The marginal and joint distributions are independent under the product "
        "measure. This is a well-known result in probability theory.\n"
    )
    submission["evidence"][0]["sha256"] = support._digest(evidence)
    support._write_json(app / "submission.json", submission)
    result = support._run_verifier(task, app, logs)
    assert result["evidence_validity"] == 0.0
    assert result["reward"] == 0.0


def test_contradictory_keyword_evidence_is_rejected(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    evidence = app / "evidence/answer.txt"
    evidence.write_text(
        "The ratio objective is unchanged, the binary domain is not relaxed, "
        "and no budget is added. Residual certificate repair words appear here, "
        "with maximum transformed residual sum zero.\n"
    )
    submission["evidence"][0]["sha256"] = support._digest(evidence)
    support._write_json(app / "submission.json", submission)
    result = support._run_verifier(task, app, logs)
    assert result["evidence_validity"] == 0.0
    assert result["reward"] == 0.0


def test_large_valid_evidence_is_accepted(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    evidence = app / "evidence/answer.txt"
    base = (
        "The public proof replaces the ratio objective, relaxes the binary "
        "domain, and adds an undeclared budget. The exact residual certificate "
        "repairs the frozen objective: every coordinate is chosen by its signed "
        "residual and the maximum transformed residual is zero.\n"
    )
    # Put the explanation beyond both the former cap and a fixed prefix scan.
    padding = "This line is additional commentary that is allowed and ignored.\n"
    evidence.write_text(padding * 20_000 + base)
    submission["evidence"][0]["sha256"] = support._digest(evidence)
    support._write_json(app / "submission.json", submission)
    result = support._run_verifier(task, app, logs)
    assert result["evidence_validity"] == 1.0
    assert result["reward"] == 1.0


def test_oversized_index_array_is_rejected_without_crashing(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["selected_indices"] = list(range(24)) * 10000
    support._bind_result_evidence(app, submission)
    support._write_json(app / "submission.json", submission)
    result = support._run_verifier(task, app, logs)
    assert result["correctness"] == 0.0
    assert result["reward"] == 0.0
