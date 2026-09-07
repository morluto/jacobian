"""Defining-invariant tests for chordal recognition."""

from __future__ import annotations

import networkx as nx
import pytest
from pydantic import ValidationError

from jacobian.math.graphs.chordal._models import (
    ChordalRecognitionRequest,
    ChordalRecognitionResult,
)
from jacobian.math.graphs.chordal.operations import recognize_chordal
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _graph(vertices: list[str], edges: list[tuple[str, str]]) -> SimpleUndirectedGraph:
    return SimpleUndirectedGraph(
        vertices=tuple(vertices),
        edges=tuple(
            (left, right) if left < right else (right, left) for left, right in edges
        ),
    )


def _adjacency(graph: SimpleUndirectedGraph) -> dict[str, set[str]]:
    neighbors: dict[str, set[str]] = {vertex: set() for vertex in graph.vertices}
    for left, right in graph.edges:
        neighbors[left].add(right)
        neighbors[right].add(left)
    return neighbors


def _assert_valid_peo(graph: SimpleUndirectedGraph, ordering: tuple[str, ...]) -> None:
    assert set(ordering) == set(graph.vertices) and len(ordering) == len(graph.vertices)
    neighbors = _adjacency(graph)
    position = {vertex: rank for rank, vertex in enumerate(ordering)}
    for rank, vertex in enumerate(ordering):
        later = [peer for peer in neighbors[vertex] if position[peer] > rank]
        for left in range(len(later)):
            for right in range(left + 1, len(later)):
                assert later[right] in neighbors[later[left]]


def _assert_induced_cycle(graph: SimpleUndirectedGraph, cycle: tuple[str, ...]) -> None:
    assert len(cycle) >= 4 and len(set(cycle)) == len(cycle)
    neighbors = _adjacency(graph)
    for first, second in zip(cycle, cycle[1:] + cycle[:1], strict=True):
        assert second in neighbors[first]
    for left in range(len(cycle)):
        for right in range(left + 1, len(cycle)):
            adjacent_positions = {left, right} in (
                {position, (position + 1) % len(cycle)}
                for position in range(len(cycle))
            )
            if not adjacent_positions:
                assert cycle[right] not in neighbors[cycle[left]]


class TestFixtures:
    def test_path_is_chordal(self) -> None:
        result = recognize_chordal(_graph(["a", "b", "c"], [("a", "b"), ("b", "c")]))
        assert result.status == "CHORDAL"
        _assert_valid_peo(result.graph, result.elimination_ordering)

    def test_c4_is_not_chordal(self) -> None:
        result = recognize_chordal(
            _graph(
                ["a", "b", "c", "d"],
                [("a", "b"), ("b", "c"), ("c", "d"), ("a", "d")],
            )
        )
        assert result.status == "NONCHORDAL"
        _assert_induced_cycle(result.graph, result.induced_cycle)

    def test_c4_plus_diagonal_is_chordal(self) -> None:
        result = recognize_chordal(
            _graph(
                ["a", "b", "c", "d"],
                [("a", "b"), ("b", "c"), ("c", "d"), ("a", "d"), ("a", "c")],
            )
        )
        assert result.status == "CHORDAL"
        _assert_valid_peo(result.graph, result.elimination_ordering)

    def test_star_is_chordal_despite_bad_candidate_order(self) -> None:
        graph = _graph(["c", "a", "b", "d"], [("c", "a"), ("c", "b"), ("c", "d")])
        result = recognize_chordal(graph)
        assert result.status == "CHORDAL"
        _assert_valid_peo(graph, result.elimination_ordering)

    def test_empty_complete_isolated_disconnected(self) -> None:
        assert recognize_chordal(_graph([], [])).status == "CHORDAL"
        assert recognize_chordal(_graph(["x"], [])).status == "CHORDAL"
        complete = _graph(
            ["a", "b", "c"],
            [("a", "b"), ("a", "c"), ("b", "c")],
        )
        assert recognize_chordal(complete).status == "CHORDAL"
        isolated = _graph(["a", "b", "c"], [("a", "b")])
        assert recognize_chordal(isolated).status == "CHORDAL"
        disconnected = _graph(
            ["a", "b", "c", "d"],
            [("a", "b"), ("c", "d")],
        )
        assert recognize_chordal(disconnected).status == "CHORDAL"
        both = _graph(
            ["a", "b", "c", "d", "e"],
            [("a", "b"), ("b", "c"), ("c", "d"), ("a", "d"), ("d", "e")],
        )
        result = recognize_chordal(both)
        assert result.status == "NONCHORDAL"
        _assert_induced_cycle(result.graph, result.induced_cycle)


class TestExhaustiveAgreement:
    @pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
    def test_all_labeled_graphs_match_networkx(self, order: int) -> None:
        vertices = [f"v{index}" for index in range(order)]
        pairs = [
            (vertices[left], vertices[right])
            for left in range(order)
            for right in range(left + 1, order)
        ]
        backend: nx.Graph[str] = nx.Graph()
        backend.add_nodes_from(vertices)
        for mask in range(1 << len(pairs)):
            chosen = [
                pairs[index] for index in range(len(pairs)) if mask & (1 << index)
            ]
            graph = _graph(vertices, chosen)
            backend.clear_edges()
            backend.add_edges_from(chosen)
            expected = nx.is_chordal(backend)
            result = recognize_chordal(graph)
            assert (result.status == "CHORDAL") == expected
            if expected:
                _assert_valid_peo(graph, result.elimination_ordering)
            else:
                _assert_induced_cycle(graph, result.induced_cycle)

    def test_relabeling_preserves_status(self) -> None:
        graph = _graph(
            ["a", "b", "c", "d"],
            [("a", "b"), ("b", "c"), ("c", "d"), ("a", "d")],
        )
        relabeled = _graph(
            ["w", "x", "y", "z"],
            [("w", "x"), ("x", "y"), ("y", "z"), ("w", "z")],
        )
        assert recognize_chordal(graph).status == recognize_chordal(relabeled).status


class TestContracts:
    def test_result_reparses(self) -> None:
        result = recognize_chordal(_graph(["a", "b", "c"], [("a", "b"), ("b", "c")]))
        assert (
            ChordalRecognitionResult.model_validate_json(result.model_dump_json())
            == result
        )

    def test_request_path_matches_native(self) -> None:
        from jacobian.math.graphs.chordal._tools import _compute_chordal_recognition

        request = ChordalRecognitionRequest(graph=_graph(["a", "b"], [("a", "b")]))
        assert _compute_chordal_recognition(request).status == "CHORDAL"

    def test_mismatched_verdict_rejected(self) -> None:
        graph = _graph(["a", "b"], [("a", "b")])
        with pytest.raises(ValidationError):
            ChordalRecognitionResult.model_validate(
                {
                    "graph": graph.model_dump(mode="json"),
                    "status": "CHORDAL",
                    "elimination_ordering": ["a", "b"],
                    "induced_cycle": ["a", "b", "a", "b"],
                }
            )
