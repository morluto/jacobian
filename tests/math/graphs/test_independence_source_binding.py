"""Regression tests binding independence-number results to their source graph."""

from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from jacobian.math.graphs import independence as independence_models
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
    assert empty.result_schema_version == "2"
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


def test_forged_nonmaximum_optimum_is_rejected_by_source_replay() -> None:
    feasible = solve_independence_number(
        IndependenceNumberRequest(graph=_graph(("a", "b", "c"), ()))
    )
    dumped = feasible.model_dump()
    dumped["graph"]["edges"] = [["a", "b"], ["b", "c"]]
    dumped["witness_vertices"] = ["a"]
    for field in (
        "optimum_value",
        "incumbent_value",
        "lower_bound",
        "upper_bound",
    ):
        dumped[field] = 1
    dumped["termination_reason"] = "OPTIMUM_ESTABLISHED"
    with pytest.raises(ValidationError, match="contradicts"):
        IndependenceNumberResult.model_validate(dumped)


def test_edge_removal_invalidating_the_claimed_optimum_is_rejected() -> None:
    triangle = solve_independence_number(
        IndependenceNumberRequest(
            graph=_graph(("a", "b", "c"), (("a", "b"), ("a", "c"), ("b", "c")))
        )
    )
    assert triangle.optimum_value == 1
    dumped = triangle.model_dump()
    dumped["graph"]["edges"] = []
    with pytest.raises(ValidationError, match="contradicts"):
        IndependenceNumberResult.model_validate(dumped)


def test_exhausted_exact_replay_budget_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = IndependenceNumberRequest(
        graph=_graph(("a", "b", "c", "d"), (("a", "b"), ("b", "c"), ("c", "d")))
    )
    result = solve_independence_number(request)
    assert result.status == "EXACT"
    monkeypatch.setattr(independence_models, "_EXACT_REPLAY_SEARCH_NODES", 1)
    with pytest.raises(ValidationError, match="was not reproduced"):
        IndependenceNumberResult.model_validate(result.model_dump())


def _matching_graph(edges: int) -> SimpleUndirectedGraph:
    return _graph(
        tuple(f"m{index:02d}" for index in range(2 * edges)),
        tuple((f"m{2 * index:02d}", f"m{2 * index + 1:02d}") for index in range(edges)),
    )


def test_matching_produced_exact_result_revalidates() -> None:
    """The reported rejection case: 15 disjoint edges, order 30, optimum 15.

    The producing solve must emit an ``EXACT`` payload whose serialized
    output loads back as an equal ``IndependenceNumberResult``, so every
    produced result revalidates against the retained source graph.
    """

    graph = _matching_graph(15)
    result = solve_independence_number(IndependenceNumberRequest(graph=graph))
    assert result.status == "EXACT"
    assert result.optimum_value == 15
    assert result.order == 30
    assert result.termination_reason == "OPTIMUM_ESTABLISHED"
    assert IndependenceNumberResult.model_validate(result.model_dump()) == result


def test_structured_disjoint_graphs_replay_within_tight_budgets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Component decomposition keeps disjoint structures inside tiny budgets.

    Ten disjoint five-cycles have optimum 20, isolated vertices are forced
    without branching, and the whole replay stays deterministic and exact
    well below the default search-node envelope that the naive search
    exceeded on a 30-vertex matching.
    """

    cycle_vertices = []
    cycle_edges = []
    for component in range(10):
        ids = [f"c{component}_{index}" for index in range(5)]
        cycle_vertices += ids
        for index in range(5):
            left, right = sorted((ids[index], ids[(index + 1) % 5]))
            cycle_edges.append((left, right))
    cycles = _graph(tuple(cycle_vertices), tuple(cycle_edges))
    mixed = _graph(
        ("i0", "i1", "i2", "p0", "p1", "p2"),
        (("p0", "p1"), ("p1", "p2")),
    )

    produced = []
    for graph, expected in ((cycles, 20), (mixed, 5), (_matching_graph(15), 15)):
        result = solve_independence_number(IndependenceNumberRequest(graph=graph))
        assert result.status == "EXACT"
        assert result.optimum_value == expected
        produced.append(result)

    monkeypatch.setattr(independence_models, "_EXACT_REPLAY_SEARCH_NODES", 512)
    for result in produced:
        assert IndependenceNumberResult.model_validate(result.model_dump()) == result


def test_unreplayable_solver_optimum_demotes_to_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A solver optimum the bounded replay cannot certify claims no optimum."""

    request = IndependenceNumberRequest(
        graph=_graph(("a", "b", "c", "d"), (("a", "b"), ("b", "c"), ("c", "d")))
    )
    monkeypatch.setattr(independence_models, "_EXACT_REPLAY_SEARCH_NODES", 1)
    result = solve_independence_number(request)
    assert result.status == "UNKNOWN"
    assert result.optimum_value is None
    assert result.termination_reason == "REPLAY_INCOMPLETE"
    assert result.lower_bound == result.incumbent_value
    assert result.upper_bound == 4
    assert IndependenceNumberResult.model_validate(result.model_dump()) == result
