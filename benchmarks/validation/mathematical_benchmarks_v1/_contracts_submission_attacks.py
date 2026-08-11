"""Generic malformed-submission and attack contract tests.

Cross-task adversarial fixtures: malformed output, missing output, wrong
result, mismatched claim, incomplete scope, escaped evidence, and unsupported
VERIFIED claims must all fail closed with the correct diagnostic dimension.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import support


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
        assert rejected.details[component] == 0.0
        assert rejected.reward < 1.0
    else:
        assert rejected.reward == 0.0
    if attack == "unsupported-verified":
        assert rejected.details["false_certification"] is True
