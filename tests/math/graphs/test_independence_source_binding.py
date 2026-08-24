"""Regression tests binding independence-number results to their source graph."""

from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from jacobian.math.graphs._independence_z3 import solve_independence_number
from jacobian.math.graphs.independence import (
    IndependenceNumberRequest,
    IndependenceNumberResult,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _graph(
    vertices: tuple[str, ...], edges: tuple[tuple[str, str], ...]
) -> SimpleUndirectedGraph:
    return SimpleUndirectedGraph(vertices=vertices, edges=edges)


def _path_graph() -> SimpleUndirectedGraph:
    return _graph(("a", "b", "c"), (("a", "b"), ("b", "c")))


def test_producer_results_replay_against_source() -> None:
    empty = solve_independence_number(IndependenceNumberRequest(graph=_graph((), ())))
    assert empty.status == "EXACT"
    assert empty.optimum_value == 0
    assert IndependenceNumberResult.model_validate(empty.model_dump()) == empty

    complete = solve_independence_number(
        IndependenceNumberRequest(graph=_graph(("a", "b"), (("a", "b"),)))
    )
    assert complete.status == "EXACT"
    assert complete.optimum_value == 1
    assert complete.witness_vertices in (("a",), ("b",))

    disconnected = solve_independence_number(
        IndependenceNumberRequest(graph=_graph(("a", "b", "c", "d"), (("a", "b"),)))
    )
    assert disconnected.optimum_value == 3

    nonunique = solve_independence_number(
        IndependenceNumberRequest(graph=_graph(("a", "b"), ()))
    )
    assert nonunique.optimum_value == 2
    assert nonunique.witness_vertices == ("a", "b")


def test_witness_membership_and_edge_freedom_are_enforced() -> None:
    request = IndependenceNumberRequest(graph=_path_graph())
    result = solve_independence_number(request)
    assert result.optimum_value == 2
    dumped = result.model_dump()

    nonexistent = copy.deepcopy(dumped)
    nonexistent["witness_vertices"] = ["a", "zz"]
    with pytest.raises(ValidationError, match="belong to the source graph"):
        IndependenceNumberResult.model_validate(nonexistent)

    added_edge = copy.deepcopy(dumped)
    added_edge["graph"]["edges"] = [
        ["a", "b"],
        ["b", "c"],
        ["a", "c"],
    ]
    with pytest.raises(ValidationError, match="both endpoints"):
        IndependenceNumberResult.model_validate(added_edge)

    foreign_graph = copy.deepcopy(dumped)
    foreign_graph["graph"] = {
        "graph_schema_version": "1",
        "vertices": ["a", "b", "c", "d"],
        "edges": [["a", "d"]],
    }
    with pytest.raises(ValidationError, match="order must match"):
        IndependenceNumberResult.model_validate(foreign_graph)


def test_field_mutations_are_rejected() -> None:
    request = IndependenceNumberRequest(graph=_path_graph())
    result = solve_independence_number(request)
    dumped = result.model_dump()

    forged_order = copy.deepcopy(dumped)
    forged_order["order"] = 7
    with pytest.raises(ValidationError, match="reported order"):
        IndependenceNumberResult.model_validate(forged_order)

    unsorted_witness = copy.deepcopy(dumped)
    if len(unsorted_witness["witness_vertices"]) > 1:
        unsorted_witness["witness_vertices"] = list(
            reversed(unsorted_witness["witness_vertices"])
        )
        with pytest.raises(ValidationError, match="canonically sorted"):
            IndependenceNumberResult.model_validate(unsorted_witness)

    cardinality_break = copy.deepcopy(dumped)
    cardinality_break["incumbent_value"] = result.incumbent_value + 1
    with pytest.raises(ValidationError, match="cardinality"):
        IndependenceNumberResult.model_validate(cardinality_break)

    bound_break = copy.deepcopy(dumped)
    bound_break["lower_bound"] = result.lower_bound + 1
    with pytest.raises(ValidationError, match="lower bound"):
        IndependenceNumberResult.model_validate(bound_break)

    false_optimum = copy.deepcopy(dumped)
    false_optimum["status"] = "UNKNOWN"
    with pytest.raises(ValidationError, match="cannot claim an optimum"):
        IndependenceNumberResult.model_validate(false_optimum)

    unknown_with_optimum = copy.deepcopy(dumped)
    unknown_with_optimum["status"] = "UNKNOWN"
    unknown_with_optimum["optimum_value"] = result.optimum_value
    with pytest.raises(ValidationError, match="cannot claim an optimum"):
        IndependenceNumberResult.model_validate(unknown_with_optimum)


def test_matching_via_edge_intersection_composition() -> None:
    """The Atlas move dual -> edge-intersection -> independence stays bound.

    The triangle hypergraphs of K4 and K5 have pairwise-intersecting
    hyperedges, so their edge-intersection graphs are complete and the
    matching numbers replay as one.
    """

    from itertools import combinations

    for order in (4, 5):
        triangles = [frozenset(c) for c in combinations(range(order), 3)]
        count = len(triangles)
        pairs = tuple(
            (f"t{i}", f"t{j}")
            for i in range(count)
            for j in range(i + 1, count)
            if triangles[i] & triangles[j]
        )
        graph = _graph(tuple(f"t{i}" for i in range(count)), pairs)
        result = solve_independence_number(IndependenceNumberRequest(graph=graph))
        assert result.status == "EXACT"
        assert result.optimum_value == 1
