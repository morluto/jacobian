from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.agent_workflow_v1 import support

TASK = "newton-polygon-factorization-audit"


def _case(tmp_path: Path):
    return support._prepare_case(tmp_path, TASK, "computed")


def _rewrite(app: Path, submission: dict) -> None:
    support._bind_result_evidence(app, submission)
    support._write_json(app / "submission.json", submission)


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
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0
