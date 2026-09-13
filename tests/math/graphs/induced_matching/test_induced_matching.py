"""Maximum induced matching tests."""

from itertools import combinations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.graphs.induced_matching._models import MaximumInducedMatchingResult
from jacobian.math.graphs.induced_matching.operations import maximum_induced_matching
from jacobian.math.graphs.values import SimpleUndirectedGraph


def test_path_five_has_two_edge_induced_matching() -> None:
    graph = SimpleUndirectedGraph(
        vertices=("0", "1", "2", "3", "4"),
        edges=(("0", "1"), ("1", "2"), ("2", "3"), ("3", "4")),
    )
    result = maximum_induced_matching(graph)
    assert result.status == "EXACT"
    assert result.cardinality == result.lower_bound == result.upper_bound == 2
    assert result.selected_edge_ids == ("e0", "e3")
    assert result.induced_endpoint_graph.edges == (("0", "1"), ("3", "4"))


def test_empty_graph_has_empty_induced_matching() -> None:
    graph = SimpleUndirectedGraph(vertices=("v",), edges=())
    result = maximum_induced_matching(graph)
    assert result.cardinality == 0
    assert result.selected_edge_ids == ()
    assert result.induced_endpoint_graph.vertices == ()


def test_source_edge_and_endpoint_axes_ignore_input_row_order() -> None:
    graph = SimpleUndirectedGraph(
        vertices=("f", "e", "d", "c", "b", "a"),
        edges=(("e", "f"), ("a", "b"), ("c", "d")),
    )
    result = maximum_induced_matching(graph)
    assert tuple(binding.edge_id for binding in result.source_edges) == (
        "e0",
        "e1",
        "e2",
    )
    assert tuple(binding.endpoints for binding in result.source_edges) == (
        ("a", "b"),
        ("c", "d"),
        ("e", "f"),
    )
    assert result.selected_edge_ids == ("e0", "e1", "e2")
    assert result.induced_endpoint_graph.vertices == (
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
    )
    assert result.induced_endpoint_graph.edges == (
        ("a", "b"),
        ("c", "d"),
        ("e", "f"),
    )


def test_lexicographically_smallest_maximum_witness_is_returned() -> None:
    # A six-vertex path has maximum induced matchings (e0,e3), (e0,e4), and
    # (e1,e4); the first edge-ID family is the canonical tie-break.
    graph = SimpleUndirectedGraph(
        vertices=tuple(str(index) for index in range(6)),
        edges=tuple((str(index), str(index + 1)) for index in range(5)),
    )
    result = maximum_induced_matching(graph)
    assert result.status == "EXACT"
    assert result.selected_edge_ids == ("e0", "e3")


def test_lexicographic_tie_break_handles_double_digit_edge_ids() -> None:
    graph = SimpleUndirectedGraph(
        vertices=tuple(str(index) for index in range(7)),
        edges=(
            ("0", "2"),
            ("0", "3"),
            ("0", "4"),
            ("1", "3"),
            ("1", "4"),
            ("1", "5"),
            ("1", "6"),
            ("2", "3"),
            ("3", "4"),
            ("3", "6"),
            ("4", "5"),
            ("5", "6"),
        ),
    )
    result = maximum_induced_matching(graph)
    assert result.cardinality == 2
    assert result.selected_edge_ids == ("e0", "e11")


def test_complete_bipartite_graph_has_one_edge_induced_matching() -> None:
    graph = SimpleUndirectedGraph(
        vertices=("a0", "a1", "b0", "b1"),
        edges=(("a0", "b0"), ("a0", "b1"), ("a1", "b0"), ("a1", "b1")),
    )
    result = maximum_induced_matching(graph)
    assert result.status == "EXACT"
    assert result.cardinality == 1
    assert result.selected_edge_ids == ("e0",)
    assert result.induced_endpoint_graph.edges == (("a0", "b0"),)


def test_cycle_and_star_have_the_expected_induced_matching_numbers() -> None:
    cycle = SimpleUndirectedGraph(
        vertices=("0", "1", "2", "3", "4"),
        edges=(
            ("0", "1"),
            ("0", "4"),
            ("1", "2"),
            ("2", "3"),
            ("3", "4"),
        ),
    )
    star = SimpleUndirectedGraph(
        vertices=("c", "a", "b", "d"),
        edges=(("a", "c"), ("b", "c"), ("c", "d")),
    )
    assert maximum_induced_matching(cycle).cardinality == 1
    assert maximum_induced_matching(star).cardinality == 1


def test_small_graph_matches_exhaustive_edge_subset_enumeration() -> None:
    graph = SimpleUndirectedGraph(
        vertices=("0", "1", "2", "3", "4"),
        edges=(
            ("0", "1"),
            ("0", "2"),
            ("1", "2"),
            ("1", "3"),
            ("2", "4"),
        ),
    )
    feasible_sizes: list[int] = []
    for size in range(len(graph.edges) + 1):
        for selected_edges in combinations(graph.edges, size):
            endpoints = {vertex for edge in selected_edges for vertex in edge}
            if len(endpoints) != 2 * size:
                continue
            induced_edges = tuple(
                edge
                for edge in graph.edges
                if edge[0] in endpoints and edge[1] in endpoints
            )
            if induced_edges == selected_edges:
                feasible_sizes.append(size)
    result = maximum_induced_matching(graph)
    assert result.status == "EXACT"
    assert result.cardinality == max(feasible_sizes)


def test_disjoint_union_additivity_and_complete_endpoint_subgraph() -> None:
    graph = SimpleUndirectedGraph(
        vertices=("a", "b", "c", "d", "e", "f"),
        edges=(("a", "b"), ("c", "d"), ("d", "e"), ("e", "f")),
    )
    result = maximum_induced_matching(graph)
    assert result.cardinality == 2
    selected = {
        binding.endpoints
        for binding in result.source_edges
        if binding.edge_id in result.selected_edge_ids
    }
    assert selected == {("a", "b"), ("c", "d")}
    assert result.induced_endpoint_graph.edges == (("a", "b"), ("c", "d"))


def test_result_rejects_incomplete_endpoint_subgraph() -> None:
    graph = SimpleUndirectedGraph(
        vertices=("a", "b", "c", "d"),
        edges=(("a", "b"), ("c", "d")),
    )
    result = maximum_induced_matching(graph)
    payload = result.model_dump(mode="python")
    payload["induced_endpoint_graph"] = {
        "vertices": ["a", "b"],
        "edges": [],
    }
    with pytest.raises(ValidationError):
        MaximumInducedMatchingResult.model_validate(payload)


def test_result_rejects_cross_edges_between_selected_edges() -> None:
    graph = SimpleUndirectedGraph(
        vertices=("a", "b", "c", "d"),
        edges=(("a", "b"), ("a", "c"), ("c", "d")),
    )
    payload = maximum_induced_matching(graph).model_dump(mode="python")
    payload["status"] = "EXACT"
    payload["selected_edge_ids"] = ["e0", "e2"]
    payload["cardinality"] = 2
    payload["lower_bound"] = 2
    payload["upper_bound"] = 2
    payload["induced_endpoint_graph"] = {
        "vertices": ["a", "b", "c", "d"],
        "edges": [["a", "b"], ["a", "c"], ["c", "d"]],
    }
    with pytest.raises(ValidationError):
        MaximumInducedMatchingResult.model_validate(payload)


def test_unknown_result_must_use_source_edge_count_as_upper_bound() -> None:
    graph = SimpleUndirectedGraph(
        vertices=("a", "b", "c", "d"),
        edges=(("a", "b"), ("c", "d")),
    )
    payload = maximum_induced_matching(graph).model_dump(mode="python")
    payload["status"] = "UNKNOWN"
    payload["selected_edge_ids"] = ["e0"]
    payload["cardinality"] = 1
    payload["lower_bound"] = 1
    payload["upper_bound"] = 1
    payload["induced_endpoint_graph"] = {
        "vertices": ["a", "b"],
        "edges": [["a", "b"]],
    }
    with pytest.raises(ValidationError):
        MaximumInducedMatchingResult.model_validate(payload)


def test_double_digit_conflict_ids_are_canonically_oriented() -> None:
    vertices = tuple(f"v{i}" for i in range(12))
    graph = SimpleUndirectedGraph(
        vertices=vertices,
        edges=tuple(("v0", f"v{i}") for i in range(1, 12)),
    )
    assert maximum_induced_matching(graph).cardinality == 1


def test_oversized_conflict_graph_is_rejected_before_expansion() -> None:
    vertices = tuple(f"v{i:03}" for i in range(130))
    graph = SimpleUndirectedGraph(
        vertices=vertices,
        edges=tuple((vertices[0], vertex) for vertex in vertices[1:]),
    )
    with pytest.raises(OperationResourceAdmissionError):
        maximum_induced_matching(graph)
