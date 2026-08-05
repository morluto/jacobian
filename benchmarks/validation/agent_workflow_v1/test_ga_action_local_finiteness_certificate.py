from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest
from benchmarks.validation.agent_workflow_v1 import support

TASK = "ga-action-local-finiteness-certificate"


def _case(tmp_path: Path):
    return support._prepare_case(tmp_path, TASK, "computed")


def _rewrite(app: Path, submission: dict) -> None:
    support._bind_result_evidence(app, submission)
    support._write_json(app / "submission.json", submission)


def _rational(value: Fraction) -> str:
    return str(value)


def test_accepts_an_alternative_scaled_basis(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    result = submission["result"]
    scales = [Fraction(2), Fraction(3), Fraction(4), Fraction(5), Fraction(6)]
    original_coordinates = [Fraction(value) for value in result["f_coordinates"]]
    for index, poly in enumerate(result["basis"]):
        poly[0]["coefficient"] = _rational(scales[index])
    result["f_coordinates"] = [
        _rational(value / scale)
        for value, scale in zip(original_coordinates, scales, strict=True)
    ]
    for row, entries in enumerate(result["action_matrix"]):
        for column, poly in enumerate(entries):
            for term in poly:
                term["coefficient"] = _rational(
                    Fraction(term["coefficient"]) * scales[column] / scales[row]
                )
    _rewrite(app, submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "corruption",
    ["singular_basis", "wrong_coordinates", "wrong_action", "false_assurance"],
)
def test_rejects_corrupted_certificates(tmp_path: Path, corruption: str) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    if corruption == "singular_basis":
        submission["result"]["basis"][4] = submission["result"]["basis"][3]
    elif corruption == "wrong_coordinates":
        submission["result"]["f_coordinates"][0] = "8"
    elif corruption == "wrong_action":
        submission["result"]["action_matrix"][0][4][0]["coefficient"] = "2"
    else:
        submission["claimed_assurance"] = "VERIFIED"
    _rewrite(app, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["reward"] == 0.0


def test_rejects_visible_input_tampering(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    source = json.loads((app / "input.json").read_text())
    source["f"][0]["coefficient"] = "2"
    support._write_json(app / "input.json", source)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0
