"""Dispatch execution tests for hypergraph independence-number search."""

import pytest
import z3  # type: ignore[import-untyped]
from tests.dispatch._support import dispatch_validation_error

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation

_TRIPLE = {
    "hypergraph": {
        "vertices": ["a", "b", "c"],
        "edges": [["triple", ["a", "b", "c"]]],
    }
}


def test_math_run_executes_independence_compute() -> None:
    result = invoke_operation(
        "hypergraph.independence_number.compute", _TRIPLE, Catalog.open()
    )
    assert result.output["status"] == "EXACT"
    assert result.output["independence_number"] == 2
    assert result.output["lower_bound"] == result.output["upper_bound"] == 2


def test_math_run_rejects_infeasible_backend_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.hypergraphs import _independence_z3

    def regressed(*_args: object) -> object:
        return z3.sat, ("a", "b", "c"), ""

    monkeypatch.setattr(_independence_z3, "_check_threshold", regressed)
    with dispatch_validation_error():
        invoke_operation(
            "hypergraph.independence_number.compute", _TRIPLE, Catalog.open()
        )


def test_math_run_projects_solver_error_when_sat_witness_misses_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.hypergraphs import _independence_z3

    def regressed(*_args: object) -> object:
        return z3.sat, ("b",), ""

    monkeypatch.setattr(_independence_z3, "_check_threshold", regressed)
    result = invoke_operation(
        "hypergraph.independence_number.compute",
        {
            "hypergraph": {
                "vertices": ["a", "b", "c", "d"],
                "edges": [
                    ["ab", ["a", "b"]],
                    ["ac", ["a", "c"]],
                    ["ad", ["a", "d"]],
                ],
            }
        },
        Catalog.open(),
    )
    assert result.output["status"] == "UNKNOWN"
    assert result.output["termination_reason"] == "SOLVER_ERROR"
    assert result.output["independence_number"] is None
    assert result.output["solver_calls"] == 1
