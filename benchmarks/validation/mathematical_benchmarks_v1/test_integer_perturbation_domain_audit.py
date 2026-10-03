import json
import shutil
from functools import partial
from pathlib import Path
from typing import Any

import pytest
from benchmarks.tooling.public_contract import check
from benchmarks.validation._source_module import load_source_module
from jsonschema import Draft202012Validator

from ._fixtures import assert_result_witness_protocol
from ._paths import TASKS
from ._verifier import _run_verifier

TASK = TASKS / "integer-perturbation-domain-audit"


def test_result_protocol(tmp_path: Path) -> None:
    assert_result_witness_protocol(tmp_path, "integer-perturbation-domain-audit")


def _submission(period: int = 4, magnitude: int = 1) -> dict[str, Any]:
    # Construct from the public constraints: two cancellations and a positive b.
    return {
        "result": {
            "semantic_status": "STRICTLY_WEAKER",
            "nat_redundancy": {
                "a_lower_bound": 0,
                "b_lower_bound": 1,
                "sum_lower_bound": 1,
                "rule": "ORDERED_ADDITION_LOWER_BOUND",
            },
            "integer_witness": {
                "period": period,
                "a_values": [magnitude] * period,
                "b_values": [-magnitude] * 2 + [magnitude] * (period - 2),
            },
        }
    }


def _check_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    submission: dict[str, Any],
    *,
    schema_valid: bool,
    correctness: float,
    input_text: str | None = None,
) -> None:
    app = tmp_path / "app"
    logs = tmp_path / "logs"
    app.mkdir()
    logs.mkdir()
    shutil.copy2(TASK / "environment/input.json", app / "input.json")
    if input_text is not None:
        (app / "input.json").write_text(input_text)
    submission_path = app / "submission.json"
    submission_path.write_text(json.dumps(submission))

    schema = json.loads((TASK / "environment/submission_schema.json").read_text())
    assert Draft202012Validator(schema).is_valid(submission) is schema_valid
    support = load_source_module(
        "_integer_perturbation_support", TASK / "tests/verifier_support.py"
    )
    monkeypatch.setattr(
        support,
        "_load_public_contract",
        partial(support._load_public_contract, TASK / "tests/public_contract.json"),
    )
    loaded = support.load_submission(submission_path, require_input_binding=False)
    assert loaded == (submission if schema_valid else None)

    result = _run_verifier(TASK, app, logs)
    assert result.details["correctness"] == correctness
    assert result.details["input_binding"] == float(input_text is None)
    assert result.reward == (correctness if input_text is None else 0.0)
    assert json.loads((logs / "reward.json").read_text()) == {"reward": result.reward}


def test_public_bounds_match_visible_domain_and_generated_files() -> None:
    source = json.loads((TASK / "environment/input.json").read_text())
    assert source["witness_contract"] == {
        "period_min": 4,
        "period_max": 8,
        "value_abs_max": 100,
        "minimum_cancellations": 2,
    }
    assert check(TASK / "tests/public_contract.json", TASK) == []


@pytest.mark.parametrize("period", range(4, 9))
@pytest.mark.parametrize("magnitude", [1, 7, 100])
def test_accepts_alternate_witnesses_across_public_domain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, period: int, magnitude: int
) -> None:
    _check_submission(
        tmp_path,
        monkeypatch,
        _submission(period, magnitude),
        schema_valid=True,
        correctness=1.0,
    )


@pytest.mark.parametrize("field", ["a_values", "b_values"])
@pytest.mark.parametrize("value", [-100, 100])
def test_scalar_endpoints_remain_structurally_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: int
) -> None:
    submission = _submission(magnitude=100)
    submission["result"]["integer_witness"][field][-1] = value
    _check_submission(
        tmp_path,
        monkeypatch,
        submission,
        schema_valid=True,
        correctness=float(field != "a_values" or value > 0),
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [("period", value) for value in [3, 9, 1000, True, 4.5, "4", None, [], {}]]
    + [
        (field, [1] * length)
        for field in ["a_values", "b_values"]
        for length in [3, 9, 1000]
    ]
    + [
        (field, value)
        for field in ["a_values", "b_values"]
        for value in [True, 4, "1,1,1,1", None, {}]
    ],
)
def test_rejects_out_of_domain_shapes_before_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    submission = _submission()
    submission["result"]["integer_witness"][field] = value
    _check_submission(
        tmp_path, monkeypatch, submission, schema_valid=False, correctness=0.0
    )


@pytest.mark.parametrize("field", ["a_values", "b_values"])
@pytest.mark.parametrize(
    "value", [-101, 101, -(10**99), 10**99, True, 1.5, "1", None, {}, []]
)
def test_rejects_out_of_domain_members_before_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    submission = _submission()
    submission["result"]["integer_witness"][field][0] = value
    _check_submission(
        tmp_path, monkeypatch, submission, schema_valid=False, correctness=0.0
    )


def test_rejects_unbounded_period_and_vectors_before_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _check_submission(
        tmp_path,
        monkeypatch,
        _submission(1000, 10**99),
        schema_valid=False,
        correctness=0.0,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("a_values", [-1, 1, 1, 1], id="negative-a"),
        pytest.param("a_values", [0, 1, 1, 1], id="zero-a"),
        pytest.param("b_values", [-1, -1, 0, 1], id="zero-b"),
        pytest.param("b_values", [1, 1, 1, 1], id="all-positive-b"),
        pytest.param("b_values", [-1, -1, -1, -1], id="all-negative-b"),
        pytest.param("b_values", [-2, -2, 1, 1], id="no-cancellations"),
        pytest.param("b_values", [-1, -2, 1, 1], id="one-cancellation"),
        pytest.param("period", 5, id="period-mismatch"),
        pytest.param("a_values", [1] * 5, id="a-length-mismatch"),
        pytest.param("b_values", [-1, -1, 1, 1, 1], id="b-length-mismatch"),
        pytest.param("period", 4.0, id="inexact-period-type"),
        pytest.param("a_values", [1.0, 1, 1, 1], id="inexact-a-type"),
        pytest.param("b_values", [-1.0, -1, 1, 1], id="inexact-b-type"),
    ],
)
def test_bounded_false_claims_reach_mathematical_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    submission = _submission()
    submission["result"]["integer_witness"][field] = value
    _check_submission(
        tmp_path, monkeypatch, submission, schema_valid=True, correctness=0.0
    )


@pytest.mark.parametrize("input_text", ["{}", "{", "[]"])
def test_input_binding_remains_independent_of_correctness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, input_text: str
) -> None:
    _check_submission(
        tmp_path,
        monkeypatch,
        _submission(),
        schema_valid=True,
        correctness=1.0,
        input_text=input_text,
    )
