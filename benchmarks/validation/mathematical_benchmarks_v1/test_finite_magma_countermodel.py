"""Public finite-carrier structure and independent countermodel replay."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from benchmarks.tooling.public_contract import check
from benchmarks.validation._source_module import load_source_module
from benchmarks.validation._verifier_child import run_verifier_in_child
from jsonschema import Draft202012Validator

from ._fixtures import assert_result_witness_protocol

TASK = (
    Path(__file__).resolve().parents[3]
    / "benchmarks/datasets/mathematical-benchmarks-v1/finite-magma-countermodel"
)
PUBLIC_INPUT = json.loads((TASK / "environment/input.json").read_text())


@pytest.fixture
def support(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    module = load_source_module(
        "_finite_magma_support", TASK / "tests/verifier_support.py"
    )
    # Redirect only the default mounted path; keep the real contract reader,
    # schema validator, and strict submission loader in the exercised path.
    monkeypatch.setattr(
        module._load_public_contract,
        "__defaults__",
        (TASK / "tests/public_contract.json",),
    )
    return module


def _submission(
    order: int,
    table: list[list[int]],
    assignment: tuple[int, int] = (0, 0),
) -> dict[str, Any]:
    return {
        "result": {
            "order": order,
            "table": table,
            "refuting_assignment": {"x": assignment[0], "y": assignment[1]},
            "premise_holds_universally": True,
            "target_holds_universally": False,
            "minimality_checked_orders": [
                n for n in PUBLIC_INPUT["search_orders"] if n < order
            ],
        }
    }


def _assert_submission(
    tmp_path: Path,
    support: ModuleType,
    submission: dict[str, Any],
    *,
    schema_valid: bool,
    reward: float,
) -> None:
    schema = json.loads((TASK / "environment/submission_schema.json").read_text())
    assert Draft202012Validator(schema).is_valid(submission) is schema_valid
    assert support.submission_matches_public_schema(submission) is schema_valid
    app, logs = tmp_path / "app", tmp_path / "logs"
    app.mkdir(parents=True)
    logs.mkdir()
    (app / "input.json").write_bytes((TASK / "environment/input.json").read_bytes())
    path = app / "submission.json"
    path.write_text(json.dumps(submission))
    loaded = support.load_submission(path, require_input_binding=False)
    assert loaded == (submission if schema_valid else None)
    replay = run_verifier_in_child(task=TASK, app=app, logs=logs)
    assert replay.reward == reward
    assert replay.details["correctness"] == reward
    assert replay.details["input_binding"] == 1.0
    assert json.loads((logs / "reward.json").read_text()) == {"reward": reward}


def test_result_witness_protocol(tmp_path: Path) -> None:
    assert_result_witness_protocol(tmp_path, "finite-magma-countermodel")


def test_generated_contract_matches_public_search_domain() -> None:
    assert check(TASK / "tests/public_contract.json", TASK) == []
    schema = json.loads((TASK / "environment/submission_schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    branches = schema["properties"]["result"]["oneOf"]
    orders = PUBLIC_INPUT["search_orders"]
    assert orders == [1, 2]
    assert [branch["properties"]["order"]["const"] for branch in branches] == orders


def test_all_public_tables_preserve_structure_and_independent_mathematics(
    tmp_path: Path, support: ModuleType
) -> None:
    """Enumerate all 17 tables directly from the two public identities."""
    table_counts: dict[int, int] = {}
    countermodels: dict[int, list[list[list[int]]]] = {}
    for order in PUBLIC_INPUT["search_orders"]:
        table_counts[order] = 0
        countermodels[order] = []
        for flat in itertools.product(range(order), repeat=order * order):
            table = [
                list(flat[row * order : (row + 1) * order]) for row in range(order)
            ]
            table_counts[order] += 1
            # Independent evaluation of the public premise and target. Do not
            # use the hidden solution or the verifier's evaluator as an oracle.
            premise_holds = all(
                table[x][y] == table[table[table[y][x]][x]][y]
                for x, y in itertools.product(range(order), repeat=2)
            )
            refutations = [
                (x, y)
                for x, y in itertools.product(range(order), repeat=2)
                if table[table[x][y]][y] != table[table[y][x]][x]
            ]
            is_countermodel = premise_holds and bool(refutations)
            if is_countermodel:
                countermodels[order].append(table)
            _assert_submission(
                tmp_path / f"order-{order}-table-{table_counts[order]}",
                support,
                _submission(order, table, refutations[0] if refutations else (0, 0)),
                schema_valid=True,
                reward=float(is_countermodel),
            )
    assert table_counts == {1: 1, 2: 16}
    assert countermodels == {1: [], 2: [[[0, 1], [0, 1]], [[1, 0], [1, 0]]]}


@pytest.mark.parametrize(
    ("order", "rows", "columns"),
    [
        (order, rows, columns)
        for order, rows, columns in itertools.product((1, 2), repeat=3)
        if (rows, columns) != (order, order)
    ],
)
def test_rejects_mixed_order_and_table_dimensions(
    tmp_path: Path, support: ModuleType, order: int, rows: int, columns: int
) -> None:
    _assert_submission(
        tmp_path,
        support,
        _submission(order, [[0] * columns for _ in range(rows)]),
        schema_valid=False,
        reward=0.0,
    )


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("field", ["cell", "x", "y"])
@pytest.mark.parametrize("boundary", ["negative", "at-order"])
def test_rejects_out_of_carrier_values(
    tmp_path: Path,
    support: ModuleType,
    order: int,
    field: str,
    boundary: str,
) -> None:
    candidate = _submission(order, [[0] * order for _ in range(order)])
    value = -1 if boundary == "negative" else order
    if field == "cell":
        candidate["result"]["table"][0][0] = value
    else:
        candidate["result"]["refuting_assignment"][field] = value
    _assert_submission(tmp_path, support, candidate, schema_valid=False, reward=0.0)


@pytest.mark.parametrize(
    ("order", "checked"),
    [(1, [1]), (1, [2]), (2, []), (2, [2]), (2, [1, 2]), (2, [1, 1])],
)
def test_rejects_inexact_minimality_orders(
    tmp_path: Path, support: ModuleType, order: int, checked: list[int]
) -> None:
    candidate = _submission(order, [[0] * order for _ in range(order)])
    candidate["result"]["minimality_checked_orders"] = checked
    _assert_submission(tmp_path, support, candidate, schema_valid=False, reward=0.0)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("result", "order"), 0),
        (("result", "order"), 3),
        (("result", "order"), True),
        (("result", "order"), 1.5),
        (("result", "order"), "2"),
        (("result", "table"), []),
        (("result", "table"), [[0, 1], [0, 1], [0, 1]]),
        (("result", "table"), [[0, 1], [0]]),
        (("result", "table"), [[0, 1], [0, 1, 0]]),
        (("result", "table", 0), True),
        (("result", "table", 0, 0), True),
        (("result", "table", 0, 0), 0.5),
        (("result", "table", 0, 0), "0"),
        (("result", "refuting_assignment", "x"), False),
        (("result", "refuting_assignment", "y"), True),
        (("result", "refuting_assignment", "x"), 0.5),
        (("result", "refuting_assignment", "y"), "1"),
        (("result", "refuting_assignment"), {"x": 0}),
        (("result", "refuting_assignment", "z"), 0),
        (("result", "minimality_checked_orders"), [True]),
        (("result", "minimality_checked_orders"), [1.5]),
        (("result", "minimality_checked_orders"), ["1"]),
        (("result", "minimality_checked_orders"), 1),
        (("result", "premise_holds_universally"), 1),
        (("result", "target_holds_universally"), 0),
        (("result", "unknown"), None),
        (("unknown",), None),
        (("result",), True),
    ],
)
def test_rejects_malformed_structure_and_types(
    tmp_path: Path,
    support: ModuleType,
    path: tuple[str | int, ...],
    value: object,
) -> None:
    candidate = _submission(2, [[0, 1], [0, 1]], (0, 1))
    target: Any = candidate
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    _assert_submission(tmp_path, support, candidate, schema_valid=False, reward=0.0)


def test_nonrefuting_assignment_is_structurally_valid_but_scores_zero(
    tmp_path: Path, support: ModuleType
) -> None:
    _assert_submission(
        tmp_path,
        support,
        _submission(2, [[0, 1], [0, 1]], (0, 0)),
        schema_valid=True,
        reward=0.0,
    )
