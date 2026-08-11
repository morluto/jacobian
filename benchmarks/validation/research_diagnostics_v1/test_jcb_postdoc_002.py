"""Public-contract regressions for the Written on the Wall II diagnostic."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from benchmarks.validation.research_diagnostics_v1 import support

TASK_NAME = "jcb-postdoc-002"


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _prepare_public_candidate(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Build a candidate from only the answer-visible task protocol."""

    task = support.DATASET / TASK_NAME
    root = tmp_path / TASK_NAME
    app = root / "app"
    logs = root / "logs"
    evidence = app / "evidence"
    evidence.mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(task / "environment" / "input.json", app / "input.json")
    answer = evidence / "answer.txt"
    answer.write_text("Public structured facts reproduced.\n", encoding="utf-8")
    support.write_json(
        app / "submission.json",
        {
            "task_id": "jacobian/jcb-postdoc-002",
            "conclusion": "COUNTEREXAMPLE_EXISTS",
            "result": {
                "expected_conclusion": "COUNTEREXAMPLE_EXISTS",
                "answer_visible_summary": "The public finite counterexample is reproduced.",
                "boundary_statement": "No extra assurance is claimed.",
                "key_facts": {
                    "tree": "7",
                    "lambda_avg": "36/7",
                    "hamiltonian_path": "absent",
                },
            },
            "claimed_assurance": "COMPUTED",
            "scope": "jacobian/jcb-postdoc-002 answer-visible diagnostic",
            "completeness": "COMPLETE",
            "evidence": [{"path": "evidence/answer.txt", "sha256": _digest(answer)}],
            "limitations": [],
        },
    )
    return task, app, logs


def test_public_structured_candidate_receives_full_reward(tmp_path: Path) -> None:
    result = support.run_verifier(*_prepare_public_candidate(tmp_path))

    assert result.reward == pytest.approx(1.0)
    assert result.details["correctness"] == 1.0
    assert result.details["assurance_calibration"] == 1.0


def test_underclaimed_assurance_reduces_aggregate_reward(tmp_path: Path) -> None:
    task, app, logs = _prepare_public_candidate(tmp_path)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    submission["claimed_assurance"] = "UNVERIFIED"
    support.write_json(submission_path, submission)

    result = support.run_verifier(task, app, logs)

    assert result.details["correctness"] == 1.0
    assert result.details["assurance_calibration"] == 0.0
    assert result.reward == pytest.approx(0.9)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("tree", 7),
        ("tree", True),
        ("lambda_avg", 36 / 7),
        ("hamiltonian_path", False),
    ),
)
def test_public_structured_facts_reject_scalar_coercion(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    task, app, logs = _prepare_public_candidate(tmp_path)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    submission["result"]["key_facts"][field] = value
    support.write_json(submission_path, submission)

    result = support.run_verifier(task, app, logs)

    assert result.reward == 0.0
    assert result.details["correctness"] == 0.0
