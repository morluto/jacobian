"""Generic evidence and artifact binding contract tests.

Cross-task invariants for evidence and verification-record binding: verifier
execution must not mutate task bundles, generated JSON evidence fixtures must
start valid, verification-record scoring must separate mathematics from
certification, canonical SAT-witness evidence must bind, and single-evidence
cardinality must be enforced.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import support


def test_verifier_execution_does_not_mutate_task_bundles(tmp_path: Path) -> None:
    before = support._task_tree_snapshot()

    result = support._run_verifier(
        *support._prepare_case(tmp_path, support.RATIONAL_TASK, "computed")
    )

    assert result["correctness"] == 1.0
    assert support._task_tree_snapshot() == before


def test_generated_json_evidence_task_fixture_starts_valid(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "inversion-aggregate-mask-audit", "computed"
    )

    evidence_path = app / "evidence" / "inversion-audit.json"
    submission = json.loads((app / "submission.json").read_text())
    assert evidence_path.is_file()
    assert submission["evidence"][0]["sha256"] == support._digest(evidence_path)
    assert support._run_verifier(task, app, logs)["reward"] == pytest.approx(1.0)


@pytest.mark.parametrize("task_name", support.VERIFICATION_RECORD_TASKS)
def test_verifier_scoring_separates_math_from_verification_record(
    tmp_path: Path,
    task_name: str,
) -> None:
    computed = support._run_verifier(
        *support._prepare_case(tmp_path, task_name, "computed")
    )
    assert computed["correctness"] == 1.0
    assert computed["reward"] == pytest.approx(0.9)
    assert computed["false_certification"] is False

    missing = support._run_verifier(
        *support._prepare_case(tmp_path, task_name, "missing")
    )
    assert missing["correctness"] == 1.0
    assert missing["reward"] == 0.0
    assert missing["false_certification"] is True

    bound = support._run_verifier(*support._prepare_case(tmp_path, task_name, "bound"))
    assert bound["correctness"] == 1.0
    assert bound["reward"] == pytest.approx(1.0)
    assert bound["false_certification"] is False

    invalid = support._run_verifier(
        *support._prepare_case(tmp_path, task_name, "invalid")
    )
    assert invalid["correctness"] == 1.0
    assert invalid["reward"] == 0.0
    assert invalid["false_certification"] is True


def test_sat_witness_canonical_verified_solution_is_bound(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, "sat-witness", "computed")
    submission = json.loads((task / "solution" / "submission.json").read_text())
    record_path = app / "evidence" / "verification-record.json"
    record_path.write_bytes(
        (task / "solution" / "verification-record.json").read_bytes()
    )
    submission["verification_record_uri"]["sha256"] = support._digest(record_path)
    support._write_json(app / "submission.json", submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["evidence_validity"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)
    assert accepted["false_certification"] is False


@pytest.mark.parametrize("task_name", support.SINGLE_EVIDENCE_TASKS)
def test_verifiers_enforce_single_evidence_cardinality(
    tmp_path: Path,
    task_name: str,
) -> None:
    task, app, logs = support._prepare_case(tmp_path, task_name, "computed")
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["evidence"].append(dict(submission["evidence"][0]))
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["evidence_validity"] == 0.0
    assert rejected["reward"] == 0.0
