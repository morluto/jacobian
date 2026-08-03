from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.agent_workflow_v1 import support

TASK = "parameterized-sharp-bound-audit"


def _case(tmp_path: Path):
    return support._prepare_case(tmp_path, TASK, "computed")


def _rewrite(app: Path, submission: dict) -> None:
    support._bind_result_evidence(app, submission)
    support._write_json(app / "submission.json", submission)


def test_accepts_permuted_certificates(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    result = submission["result"]
    result["certificate"]["tangent_variables"] = ["c", "a", "b"]
    result["certificate"]["schur_ordering"] = ["b", "c", "a"]
    result["boundary_family"] = {
        "vanishing_variable": "a",
        "other_variables": ["c", "b"],
        "parameter": "t->0+",
        "limit": "1/4",
        "attained_for_positive_parameter": False,
    }
    result["audit"]["defects"].reverse()
    _rewrite(app, submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("transition", {"numerator": 4, "denominator": 1}),
        (
            "high_regime",
            {
                "condition": "d>=15/4",
                "bound": "1/4",
                "remainder_coefficient": "d-15/4",
                "attainment": "ATTAINED_FOR_ALL_D",
            },
        ),
        (
            "boundary_family",
            {
                "vanishing_variable": "c",
                "other_variables": ["a", "b"],
                "parameter": "t->0+",
                "limit": "1/4",
                "attained_for_positive_parameter": True,
            },
        ),
    ],
)
def test_rejects_corrupted_sharpness(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"][field] = replacement
    _rewrite(app, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_evidence_binds_boundary_family(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["boundary_family"]["vanishing_variable"] = "a"
    submission["result"]["boundary_family"]["other_variables"] = ["b", "c"]
    support._bind_result_evidence(app, submission)
    evidence_path = app / "evidence" / "answer.txt"
    text = evidence_path.read_text().replace(
        'BOUNDARY_FAMILY_JSON: {"attained_for_positive_parameter":false,"limit":"1/4","other_variables":["b","c"],"parameter":"t->0+","vanishing_variable":"a"}',
        'BOUNDARY_FAMILY_JSON: {"attained_for_positive_parameter":false,"limit":"1/4","other_variables":["a","b"],"parameter":"t->0+","vanishing_variable":"c"}',
    )
    evidence_path.write_text(text)
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 1.0
    assert rejected["evidence_validity"] == 0.0
