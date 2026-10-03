from __future__ import annotations

import itertools
import json
import pathlib
import shutil
from pathlib import Path
from typing import Any

import pytest
from benchmarks.tooling.public_contract import check
from benchmarks.validation._source_module import load_source_module
from benchmarks.validation._verifier_child import (
    _MappedPathFactory,
    run_verifier_in_child,
)
from jsonschema import Draft202012Validator

from ._fixtures import assert_result_witness_protocol
from ._paths import TASKS

TASK = TASKS / "erdos-gallai-realization-audit"
# Independently enumerated labeled graphs with degrees [4, 4, 3, 3, 3, 1].
GRAPHS = (
    ((0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (4, 5)),
    ((0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 4), (3, 5)),
)


def _rows(graph: int = 0, offset: int = 0) -> list[dict[str, object]]:
    return [
        {
            "case_id": "constructive",
            "status": "GRAPHICAL",
            "edges": [[u + offset, v + offset] for u, v in GRAPHS[graph]],
            "violations": [],
        },
        {
            "case_id": "obstruction",
            "status": "NONGRAPHICAL",
            "edges": [],
            "violations": [{"k": 3, "lhs": 13, "rhs": 11}],
        },
    ]


def _schema() -> dict[str, Any]:
    schema: dict[str, Any] = json.loads(
        (TASK / "environment/submission_schema.json").read_text()
    )
    return schema


def _public_case_ids() -> list[str]:
    public_input = json.loads((TASK / "environment/input.json").read_text())
    identifiers: list[str] = [case["case_id"] for case in public_input["cases"]]
    assert all(isinstance(identifier, str) for identifier in identifiers)
    return identifiers


def _assert_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rows: list[dict[str, object]],
    *,
    schema_valid: bool,
    correct: bool,
) -> None:
    submission = {"result": {"cases": rows}}
    assert Draft202012Validator(_schema()).is_valid(submission) is schema_valid
    app, logs = tmp_path / "app", tmp_path / "logs"
    app.mkdir()
    logs.mkdir()
    shutil.copy2(TASK / "environment/input.json", app / "input.json")
    (app / "submission.json").write_text(json.dumps(submission))

    mapper = _MappedPathFactory(
        {"/app": app, "/tests": TASK / "tests", "/logs/verifier": logs}
    )
    with monkeypatch.context() as paths:
        paths.setattr(pathlib, "Path", mapper)
        support = load_source_module(
            "_erdos_gallai_identity_support", TASK / "tests/verifier_support.py"
        )
        loaded = support.load_submission()
    assert loaded == (submission if schema_valid else None)

    outcome = run_verifier_in_child(task=TASK, app=app, logs=logs)
    assert outcome.reward == float(correct)
    assert outcome.details["correctness"] == float(correct)
    assert json.loads((logs / "reward.json").read_text()) == {"reward": float(correct)}


def _assert_public_case_domain(identifiers: list[str]) -> None:
    schema = _schema()
    declared = schema["$defs"]["case"]["properties"]["case_id"]["enum"]
    cases = schema["properties"]["result"]["properties"]["cases"]
    assert len(set(identifiers)) == len(identifiers)
    assert set(declared) == set(identifiers)
    assert len(declared) == cases["minItems"] == cases["maxItems"] == len(identifiers)
    clauses = cases["allOf"]
    assert len(clauses) == len(identifiers)
    assert {
        clause["contains"]["properties"]["case_id"]["const"] for clause in clauses
    } == set(identifiers)
    for clause in clauses:
        assert clause["minContains"] == clause["maxContains"] == 1
        assert clause["contains"]["type"] == "object"
        assert clause["contains"]["required"] == ["case_id"]


def test_result_witness_protocol(tmp_path: Path) -> None:
    assert_result_witness_protocol(tmp_path, "erdos-gallai-realization-audit")


@pytest.mark.parametrize(
    "identifiers",
    tuple(itertools.product(("constructive", "obstruction", "unknown"), repeat=2)),
)
def test_all_two_row_case_id_assignments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    identifiers: tuple[str, str],
) -> None:
    rows = _rows()
    for row, identifier in zip(rows, identifiers, strict=True):
        row["case_id"] = identifier
    _assert_replay(
        tmp_path,
        monkeypatch,
        rows,
        schema_valid=set(identifiers) == set(_public_case_ids()),
        correct=identifiers == ("constructive", "obstruction"),
    )


@pytest.mark.parametrize("identifier", [None, True, 1, "", [], {}])
def test_rejects_malformed_case_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, identifier: object
) -> None:
    rows = _rows()
    rows[0]["case_id"] = identifier
    _assert_replay(tmp_path, monkeypatch, rows, schema_valid=False, correct=False)


@pytest.mark.parametrize(
    "mutation", ["duplicate-row", "missing-row", "extra-row", "missing-id"]
)
def test_rejects_missing_or_extra_case_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    rows = _rows()
    if mutation == "duplicate-row":
        rows[1] = dict(rows[0])
    elif mutation == "missing-row":
        rows.pop()
    elif mutation == "extra-row":
        rows.append(dict(rows[0]))
    else:
        rows[0].pop("case_id")
    _assert_replay(tmp_path, monkeypatch, rows, schema_valid=False, correct=False)


@pytest.mark.parametrize("graph", range(len(GRAPHS)))
@pytest.mark.parametrize("offset", [0, 1])
@pytest.mark.parametrize("reverse", [False, True])
def test_accepts_alternate_graphs_row_orders_and_label_bases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    graph: int,
    offset: int,
    reverse: bool,
) -> None:
    rows = _rows(graph, offset)
    if reverse:
        rows.reverse()
    _assert_replay(tmp_path, monkeypatch, rows, schema_valid=True, correct=True)


@pytest.mark.parametrize(
    "mutation", ["wrong-degrees", "wrong-inequality", "wrong-status"]
)
def test_mathematical_errors_remain_schema_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    rows = _rows()
    if mutation == "wrong-degrees":
        rows[0]["edges"] = [list(edge) for edge in GRAPHS[0][:-1]]
    elif mutation == "wrong-inequality":
        rows[1]["violations"] = [{"k": 3, "lhs": 13, "rhs": 12}]
    else:
        rows[0]["status"] = "NONGRAPHICAL"
    _assert_replay(tmp_path, monkeypatch, rows, schema_valid=True, correct=False)


def test_public_case_domain_matches_input_independent_of_order() -> None:
    identifiers = _public_case_ids()
    _assert_public_case_domain(identifiers)
    _assert_public_case_domain(list(reversed(identifiers)))


@pytest.mark.parametrize("mutation", ["added", "removed", "renamed"])
def test_public_case_domain_detects_input_drift(mutation: str) -> None:
    identifiers = _public_case_ids()
    if mutation == "added":
        identifiers.append("new-public-case")
    elif mutation == "removed":
        identifiers.pop()
    else:
        identifiers[0] = "renamed-public-case"
    with pytest.raises(AssertionError):
        _assert_public_case_domain(identifiers)


def test_generated_public_contract_artifacts_are_current() -> None:
    assert check(TASK / "tests/public_contract.json", TASK) == []
