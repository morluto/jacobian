from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from benchmarks.tooling.public_contract import check
from benchmarks.validation._source_module import load_source_module
from benchmarks.validation._verifier_child import run_verifier_in_child
from benchmarks.validation.mathematical_benchmarks_v1 import (
    _fixtures,
    _verifier,
)
from jsonschema import Draft202012Validator

TASK = "calendar-good-days-audit"


def test_rejects_corrupted_count(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["count"] = 15
    _fixtures._write_json(app / "submission.json", submission)
    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_tampered_input_is_a_hard_gate_without_erasing_math_diagnostic(
    tmp_path: Path,
) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    input_data = json.loads((app / "input.json").read_text())
    input_data["task_id"] = "tampered"
    _fixtures._write_json(app / "input.json", input_data)

    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 1.0
    assert rejected.reward == 0.0


# These witnesses and feasibility bounds use only the agent-visible calendar.
# They do not depend on the hidden successful output or its cardinality.
TASK_ROOT = (
    Path(__file__).resolve().parents[3]
    / "benchmarks/datasets/mathematical-benchmarks-v1"
    / TASK
)
PUBLIC_INPUT = json.loads((TASK_ROOT / "environment/input.json").read_text())
ALL_DATES = [
    {
        "month": item["month"],
        "day": day,
        "concatenated": item["month"] * 10 ** len(str(day)) + day,
    }
    for item in PUBLIC_INPUT["months"]
    for day in range(1, item["days"] + 1)
]


def _public_solution() -> dict[str, Any]:
    # Integer remainders independently implement the public divisibility rule.
    dates = [
        date
        for date in ALL_DATES
        if divmod(date["concatenated"], date["month"])[1] == 0
        and divmod(date["concatenated"], date["day"])[1] == 0
    ]
    return {"result": {"count": len(dates), "good_dates": dates}}


@pytest.fixture
def support(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    module = load_source_module(
        "_calendar_bounds_support", TASK_ROOT / "tests/verifier_support.py"
    )
    monkeypatch.setattr(
        module._load_public_contract,
        "__defaults__",
        (TASK_ROOT / "tests/public_contract.json",),
    )
    return module


def _schema_valid(submission: dict[str, Any]) -> bool:
    schema = json.loads((TASK_ROOT / "environment/submission_schema.json").read_text())
    return Draft202012Validator(schema).is_valid(submission)


def _assert_replay(
    tmp_path: Path,
    support: ModuleType,
    submission: dict[str, Any],
    *,
    schema_valid: bool,
    reward: float,
) -> None:
    assert _schema_valid(submission) is schema_valid
    assert support.submission_matches_public_schema(submission) is schema_valid
    app, logs = tmp_path / "app", tmp_path / "logs"
    app.mkdir()
    logs.mkdir()
    (app / "input.json").write_bytes(
        (TASK_ROOT / "environment/input.json").read_bytes()
    )
    path = app / "submission.json"
    path.write_text(json.dumps(submission))
    assert support.load_submission(path, require_input_binding=False) == (
        submission if schema_valid else None
    )
    replay = run_verifier_in_child(task=TASK_ROOT, app=app, logs=logs)
    assert replay.reward == reward
    assert replay.details["correctness"] == reward
    assert replay.details["input_binding"] == 1.0
    assert json.loads((logs / "reward.json").read_text()) == {"reward": reward}


def test_every_public_date_remains_structurally_licensed(support: ModuleType) -> None:
    assert len(ALL_DATES) == 92
    for date in ALL_DATES:
        submission = {"result": {"count": 1, "good_dates": [date]}}
        assert _schema_valid(submission)
        assert support.submission_matches_public_schema(submission)
    assert check(TASK_ROOT / "tests/public_contract.json", TASK_ROOT) == []


def test_independent_public_solution_is_accepted(
    tmp_path: Path, support: ModuleType
) -> None:
    _assert_replay(tmp_path, support, _public_solution(), schema_valid=True, reward=1.0)


@pytest.mark.parametrize(
    "mutation",
    ["all-dates", "empty", "count", "reverse", "missing", "concatenation", "duplicate"],
)
def test_false_mathematics_within_public_bounds_scores_zero(
    tmp_path: Path, support: ModuleType, mutation: str
) -> None:
    candidate = _public_solution()
    result = candidate["result"]
    if mutation == "all-dates":
        result.update(count=len(ALL_DATES), good_dates=ALL_DATES)
    elif mutation == "empty":
        result.update(count=0, good_dates=[])
    elif mutation == "count":
        result["count"] = 0
    elif mutation == "reverse":
        result["good_dates"] = list(reversed(result["good_dates"]))
    elif mutation == "missing":
        result["good_dates"] = result["good_dates"][1:]
        result["count"] -= 1
    elif mutation == "concatenation":
        result["good_dates"][0] = {**result["good_dates"][0], "concatenated": 32}
    else:
        result["good_dates"] = result["good_dates"] + [result["good_dates"][0]]
        result["count"] += 1
    _assert_replay(tmp_path, support, candidate, schema_valid=True, reward=0.0)


@pytest.mark.parametrize(
    "mutation",
    ["row-overflow", "count-overflow", "count-negative", "huge-count", "huge-value"],
)
def test_public_size_and_scalar_bounds_precede_replay(
    tmp_path: Path, support: ModuleType, mutation: str
) -> None:
    candidate = _public_solution()
    result = candidate["result"]
    if mutation == "row-overflow":
        result["good_dates"] = [ALL_DATES[0]] * (len(ALL_DATES) + 1)
    elif mutation == "count-overflow":
        result["count"] = len(ALL_DATES) + 1
    elif mutation == "count-negative":
        result["count"] = -1
    elif mutation == "huge-count":
        result["count"] = 10**1000
    else:
        result["good_dates"][0] = {**ALL_DATES[0], "concatenated": 10**1000}
    _assert_replay(tmp_path, support, candidate, schema_valid=False, reward=0.0)


@pytest.mark.parametrize("month", [3, 4, 5])
@pytest.mark.parametrize(
    "boundary", ["day-zero", "day-overflow", "value-below", "value-above"]
)
def test_month_specific_boundaries(
    tmp_path: Path, support: ModuleType, month: int, boundary: str
) -> None:
    dates = [date for date in ALL_DATES if date["month"] == month]
    row = dict(dates[0])
    if boundary == "day-zero":
        row["day"] = 0
    elif boundary == "day-overflow":
        row["day"] = dates[-1]["day"] + 1
    elif boundary == "value-below":
        row["concatenated"] = min(date["concatenated"] for date in dates) - 1
    else:
        row["concatenated"] = max(date["concatenated"] for date in dates) + 1
    _assert_replay(
        tmp_path,
        support,
        {"result": {"count": 1, "good_dates": [row]}},
        schema_valid=False,
        reward=0.0,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("month", 999),
        ("month", 2),
        ("month", 6),
        ("month", True),
        ("day", False),
        ("day", 1.5),
        ("concatenated", "31"),
        ("concatenated", True),
    ],
)
def test_invalid_calendar_members_and_types(
    tmp_path: Path, support: ModuleType, field: str, value: object
) -> None:
    row: dict[str, Any] = dict(ALL_DATES[0])
    row[field] = value
    _assert_replay(
        tmp_path,
        support,
        {"result": {"count": 1, "good_dates": [row]}},
        schema_valid=False,
        reward=0.0,
    )
