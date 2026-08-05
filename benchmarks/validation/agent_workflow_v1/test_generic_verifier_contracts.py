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


def test_rational_solution_rejects_unsupported_verified_claim(
    tmp_path: Path,
) -> None:
    computed = support._run_verifier(
        *support._prepare_case(tmp_path, support.RATIONAL_TASK, "computed")
    )
    assert computed["correctness"] == 1.0
    assert computed["reward"] == pytest.approx(1.0)
    assert computed["false_certification"] is False

    for scenario, expected_correctness, false_certification in (
        ("missing", 1.0, True),
        ("invalid", 0.0, False),
    ):
        result = support._run_verifier(
            *support._prepare_case(tmp_path, support.RATIONAL_TASK, scenario)
        )
        assert result["correctness"] == expected_correctness
        assert result["reward"] == 0.0
        assert result["false_certification"] is false_certification


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


@pytest.mark.parametrize("task_name", support.VERIFIER_TASKS)
def test_verifiers_reject_replaced_workspace_inputs(
    tmp_path: Path,
    task_name: str,
) -> None:
    task, app, logs = support._prepare_case(tmp_path, task_name, "computed")
    input_path = app / "input.json"
    input_data = json.loads(input_path.read_text())
    input_data["task_id"] = "tampered"
    support._write_json(input_path, input_data)

    rejected = support._run_verifier(task, app, logs)
    if task_name in support.INPUT_BINDING_DECOUPLED_TASKS:
        assert rejected["correctness"] == 1.0
        assert rejected["input_binding"] == 0.0
    else:
        assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


@pytest.mark.parametrize("task_name", support.VERIFIER_TASKS)
@pytest.mark.parametrize(
    "replacement", ("{", "[]"), ids=("invalid-json", "wrong-shape")
)
def test_verifiers_fail_closed_on_malformed_workspace_inputs(
    tmp_path: Path,
    task_name: str,
    replacement: str,
) -> None:
    task, app, logs = support._prepare_case(tmp_path, task_name, "computed")
    (app / "input.json").write_text(replacement)

    rejected = support._run_verifier(task, app, logs)
    if task_name in support.INPUT_BINDING_DECOUPLED_TASKS:
        # Mathematical correctness is reported independently of input binding;
        # the result is still canonical, so correctness stays 1.0 while the
        # separate input_binding diagnostic captures the tamper.
        assert rejected["correctness"] == 1.0
        assert rejected["input_binding"] == 0.0
    else:
        assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_verifier_rejects_symlinked_workspace_input(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, support.RATIONAL_TASK, "computed")
    input_path = app / "input.json"
    input_path.unlink()
    frozen_input = next((task / "tests").glob("*input*.json"))
    input_path.symlink_to(frozen_input)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


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
    expected_scope = (
        1.0 if task_name in support.SCOPE_INDEPENDENT_ASSURANCE_TASKS else 0.0
    )
    assert rejected["scope_accuracy"] == expected_scope
    assert rejected["reward"] == 0.0
    assert rejected["false_certification"] is False


@pytest.mark.parametrize("task_name", support.SCOPE_INDEPENDENT_ASSURANCE_TASKS)
def test_scope_independent_verifiers_preserve_scope_for_unsupported_assurance(
    tmp_path: Path,
    task_name: str,
) -> None:
    task, app, logs = support._prepare_case(tmp_path, task_name, "computed")
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["claimed_assurance"] = "CHECKED"
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["scope_accuracy"] == 1.0
    assert rejected["assurance_calibration"] == 0.0
    assert rejected["reward"] == 0.0


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
