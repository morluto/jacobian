from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.agent_workflow_v1 import support

TASK = "quartic-chebotarev-proof-repair"


def _case(tmp_path: Path):
    return support._prepare_case(tmp_path, TASK, "computed")


def _rewrite(app: Path, submission: dict) -> None:
    support._bind_result_evidence(app, submission)
    support._write_json(app / "submission.json", submission)


def test_accepts_alternative_unramified_factorization(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    # f=(x+2)(x^3-2x^2-x-2) modulo 5.
    submission["result"]["frobenius_factorization"] = {
        "prime": 5,
        "linear": [2, 1],
        "cubic": [-2, -1, -2, 1],
    }
    _rewrite(app, submission)
    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("discriminant", 4352),
        (
            "cycle_count",
            {
                "identity": 1,
                "transposition": 6,
                "three_cycle": 8,
                "fixed_point_free": 6,
                "with_fixed_point": 18,
                "group_order": 24,
            },
        ),
        ("density", [3, 4]),
    ],
)
def test_rejects_corrupted_proof_components(
    tmp_path: Path, field: str, replacement: object
) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"][field] = replacement
    _rewrite(app, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_rejects_reducible_cubic_factor(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["frobenius_factorization"] = {
        "prime": 3,
        "linear": [1, 1],
        "cubic": [0, 0, 0, 1],
    }
    _rewrite(app, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
