from __future__ import annotations

import json
from pathlib import Path

from benchmarks.validation.agent_workflow_v1 import support

TASK = "fractional-ratio-proof-repair"


def _run(tmp_path: Path, mutate=None):
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    if mutate:
        mutate(submission)
        support._bind_result_evidence(app, submission)
        support._write_json(app / "submission.json", submission)
    return support._run_verifier(task, app, logs)


def test_oracle_passes(tmp_path: Path) -> None:
    assert _run(tmp_path)["reward"] == 1.0


def test_plain_digest_bound_evidence_needs_no_private_marker(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    evidence = app / "evidence/answer.txt"
    evidence.write_text("Residual certificate supplied in submission.json.\n")
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
