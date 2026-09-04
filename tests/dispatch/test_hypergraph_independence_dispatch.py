"""Dispatch execution tests for hypergraph independence-number search."""

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


def test_math_run_admits_carrier_sized_edgeless_source() -> None:
    vertices = [f"v{i:03}" for i in reversed(range(256))]
    result = invoke_operation(
        "hypergraph.independence_number.compute",
        {"hypergraph": {"vertices": vertices, "edges": []}},
        Catalog.open(),
    )
    assert result.output["independence_number"] == 256
    assert result.output["incumbent_vertices"] == vertices
    assert result.output["solver_calls"] == 0
