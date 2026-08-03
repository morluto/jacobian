from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.agent_workflow_v1 import support

# Small fixed samples for the RESULT_JSON binding policy. Do not auto-discover
# every RESULT_JSON task: many verifiers still require task-specific prose, and a
# growing exclude list recreates the shared-edit hotspot this cleanup removes.
_KEYWORD_ONLY_TASKS = (
    "autoformalization-semantic-audit",
    "calendar-good-days-audit",
    "complex-power-sum-elimination",
    "distinct-sum-pairing-optimum",
    "modular-cubic-obstruction",
    "random-function-expectation-audit",
    "well-total-domination-counterexample",
)

_NO_MARKER_CASES = (
    ("calendar-good-days-audit", None),
    ("distinct-sum-pairing-optimum", None),
    ("modular-cubic-obstruction", None),
    ("lagrangian-projection-proof-audit", None),
    ("research-status-evidence-audit", 0.2),
    ("putnam-2adic-induction-audit", None),
)

_EMPTY_PROSE_TASKS = (
    "lean-transitive-axiom-audit",
    "putnam-2adic-induction-audit",
)

_MISMATCHED_MARKER_CASES = (
    ("metric-tsp-proof-repair", ("weights", "optimal"), 999),
    ("divisibility-construction-witness", ("a",), 999),
)


@pytest.mark.parametrize("task_name", _KEYWORD_ONLY_TASKS)
def test_keyword_only_evidence_is_accepted_with_bound_result(
    tmp_path: Path,
    task_name: str,
) -> None:
    """Keyword-only prose is accepted once RESULT_JSON binds the structured result."""
    task, app, logs = support._prepare_case(tmp_path, task_name, "computed")
    submission = json.loads((app / "submission.json").read_text())
    (app / "evidence" / "answer.txt").write_text(
        "Brief explanation.\nRESULT_JSON: {}\n"
    )
    support._bind_result_evidence(app, submission)
    support._write_json(app / "submission.json", submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["evidence_validity"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


@pytest.mark.parametrize(("task_name", "expected_reward"), _NO_MARKER_CASES)
def test_evidence_without_result_marker_is_rejected(
    tmp_path: Path,
    task_name: str,
    expected_reward: float | None,
) -> None:
    """Evidence lacking a RESULT_JSON marker must fail evidence validation."""
    task, app, logs = support._prepare_case(tmp_path, task_name, "computed")
    submission = json.loads((app / "submission.json").read_text())
    evidence_path = app / "evidence" / "answer.txt"
    evidence_path.write_text(
        "A complete explanation with all the right ideas but no structured marker.\n"
    )
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(app / "submission.json", submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 1.0
    assert rejected["evidence_validity"] == 0.0
    if expected_reward is not None:
        assert rejected["reward"] == pytest.approx(expected_reward)


@pytest.mark.parametrize("task_name", _EMPTY_PROSE_TASKS)
def test_evidence_rejects_empty_prose(tmp_path: Path, task_name: str) -> None:
    task, app, logs = support._prepare_case(tmp_path, task_name, "computed")
    submission = json.loads((app / "submission.json").read_text())
    evidence_path = app / "evidence" / "answer.txt"
    evidence_path.write_text("\n")
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(app / "submission.json", submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["evidence_validity"] == 0.0


@pytest.mark.parametrize(
    ("task_name", "result_path", "replacement"),
    _MISMATCHED_MARKER_CASES,
)
def test_evidence_rejects_mismatched_result_marker(
    tmp_path: Path,
    task_name: str,
    result_path: tuple[str, ...],
    replacement: object,
) -> None:
    """Evidence must bind the exact structured result, not just prose."""
    task, app, logs = support._prepare_case(tmp_path, task_name, "computed")
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    (app / "evidence" / "answer.txt").write_text(
        "Brief explanation.\nRESULT_JSON: {}\n"
    )
    support._bind_result_evidence(app, submission)
    tampered = json.loads(submission_path.read_text())
    target = tampered["result"]
    for key in result_path[:-1]:
        target = target[key]
    target[result_path[-1]] = replacement
    support._write_json(submission_path, tampered)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0
