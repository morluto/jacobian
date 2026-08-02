from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.agent_workflow_v1 import support


def test_verifier_execution_does_not_mutate_task_bundles(tmp_path: Path) -> None:
    before = support._task_tree_snapshot()

    result = support._run_verifier(
        *support._prepare_case(tmp_path, support.RATIONAL_TASK, "computed")
    )

    assert result["correctness"] == 1.0
    assert support._task_tree_snapshot() == before


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


def test_rational_solution_rejects_unsupported_verified_claim(
    tmp_path: Path,
) -> None:
    computed = support._run_verifier(
        *support._prepare_case(tmp_path, support.RATIONAL_TASK, "computed")
    )
    assert computed["correctness"] == 1.0
    assert computed["reward"] == pytest.approx(1.0)
    assert computed["false_certification"] is False

    for scenario in ("missing", "invalid"):
        result = support._run_verifier(
            *support._prepare_case(tmp_path, support.RATIONAL_TASK, scenario)
        )
        assert result["correctness"] == 1.0
        assert result["reward"] == 0.0
        assert result["false_certification"] is True


@pytest.mark.parametrize("task_name", support.RESOURCE_DERIVED_TASKS)
def test_resource_derived_oracles_and_assurance_boundary(
    tmp_path: Path,
    task_name: str,
) -> None:
    computed = support._run_verifier(
        *support._prepare_case(tmp_path, task_name, "computed")
    )
    assert computed["correctness"] == 1.0
    assert computed["reward"] == pytest.approx(1.0)
    assert computed["false_certification"] is False

    unsupported = support._run_verifier(
        *support._prepare_case(tmp_path, task_name, "missing")
    )
    assert unsupported["correctness"] == 1.0
    assert unsupported["reward"] == 0.0
    assert unsupported["false_certification"] is True


@pytest.mark.parametrize(
    ("task_name", "mutate"),
    [
        (
            "autoformalization-semantic-audit",
            lambda result: result["operator_mismatch_certificate"].update(
                dot_product=1
            ),
        ),
        (
            "calendar-good-days-audit",
            lambda result: result.update(count=15),
        ),
        (
            "finite-magma-countermodel",
            lambda result: result["table"][1].__setitem__(1, 2),
        ),
        (
            "gaussian-moment-generality-audit",
            lambda result: result["v_coefficients"].__setitem__(2, "-1/3"),
        ),
        (
            "matrix-square-zero-counterexample",
            lambda result: result.update(matrix=[[1, 0], [0, 0]]),
        ),
        (
            "metric-tsp-proof-repair",
            lambda result: result["weights"].update(optimal=31),
        ),
        (
            "polynomial-tail-counterexample",
            lambda result: result.update(x2="1"),
        ),
        (
            "subspace-direct-sum-counterexample",
            lambda result: result.update(dependence_coefficients=[1, 1, 1, 1]),
        ),
        (
            "well-total-domination-counterexample",
            lambda result: result.update(degree_sum=7),
        ),
        (
            "log-exponent-recovery",
            lambda result: result.update(value=59),
        ),
        (
            "random-function-expectation-audit",
            lambda result: result.update(expected_value="2025"),
        ),
        (
            "lagrangian-projection-proof-audit",
            lambda result: result["lagrangian_defect"][0].__setitem__(1, "0"),
        ),
    ],
)
def test_resource_derived_verifiers_reject_corrupted_witnesses(
    tmp_path: Path,
    task_name: str,
    mutate,
) -> None:
    task, app, logs = support._prepare_case(tmp_path, task_name, "computed")
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    mutate(submission["result"])
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


@pytest.mark.parametrize("task_name", support.VERIFIER_TASKS)
def test_verifiers_reject_unhashable_assurance(
    tmp_path: Path,
    task_name: str,
) -> None:
    task, app, logs = support._prepare_case(tmp_path, task_name, "computed")
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["claimed_assurance"] = []
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["scope_accuracy"] == 0.0
    assert rejected["reward"] == 0.0
    assert rejected["false_certification"] is False


@pytest.mark.parametrize("task_name", support.VERIFIER_TASKS)
@pytest.mark.parametrize(
    "attack",
    (
        "malformed-output",
        "missing-output",
        "wrong-result",
        "mismatched-claim",
        "incomplete-scope",
        "escaped-evidence",
        "unsupported-verified",
    ),
)
def test_verifiers_fail_closed_on_submission_attacks(
    tmp_path: Path,
    task_name: str,
    attack: str,
) -> None:
    task, app, logs = support._prepare_case(tmp_path, task_name, f"attack-{attack}")
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    if attack != "unsupported-verified":
        submission["claimed_assurance"] = "COMPUTED"

    if attack == "malformed-output":
        submission_path.write_text("{", encoding="utf-8")
    elif attack == "missing-output":
        submission_path.unlink()
    elif attack == "wrong-result":
        submission["result"] = {}
        support._write_json(submission_path, submission)
    elif attack == "mismatched-claim":
        submission["conclusion"] = "UNSUPPORTED"
        support._write_json(submission_path, submission)
    elif attack == "incomplete-scope":
        submission["scope"] = "incomplete"
        support._write_json(submission_path, submission)
    elif attack == "escaped-evidence":
        submission["evidence"] = [
            {
                "path": "../answer.txt",
                "sha256": submission["evidence"][0]["sha256"],
            }
        ]
        support._write_json(submission_path, submission)
    else:
        support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    if attack in {"incomplete-scope", "escaped-evidence"}:
        component = (
            "scope_accuracy" if attack == "incomplete-scope" else "evidence_validity"
        )
        assert rejected[component] == 0.0
        assert rejected["reward"] < 1.0
    else:
        assert rejected["reward"] == 0.0
    if attack == "unsupported-verified":
        assert rejected["false_certification"] is True
