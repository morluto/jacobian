"""Edge case tests for the simple-graph canonical value validators."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.canonical import encode_strict_json
from jacobian.math.graphs.values import (
    IndexedSimpleUndirectedGraph,
    SimpleUndirectedGraph,
    simple_undirected_graph_wire_bytes,
)


def test_graph_rejects_non_nfc_vertices() -> None:
    with pytest.raises(ValidationError):
        SimpleUndirectedGraph(vertices=("e\u0301",), edges=())


def test_graph_rejects_duplicate_vertices() -> None:
    with pytest.raises(ValidationError):
        SimpleUndirectedGraph(vertices=("a", "a"), edges=())


def test_graph_rejects_edges_with_undeclared_vertices() -> None:
    with pytest.raises(ValidationError):
        SimpleUndirectedGraph(vertices=("a",), edges=(("a", "b"),))


def test_graph_rejects_edges_out_of_order() -> None:
    with pytest.raises(ValidationError):
        SimpleUndirectedGraph(vertices=("a", "b"), edges=(("b", "a"),))


def test_graph_rejects_duplicate_edges() -> None:
    with pytest.raises(ValidationError):
        SimpleUndirectedGraph(
            vertices=("a", "b"),
            edges=(("a", "b"), ("a", "b")),
        )


def test_graph_wire_size_matches_its_canonical_value_serialization() -> None:
    graph = SimpleUndirectedGraph(vertices=("a", "é"), edges=(("a", "é"),))

    assert simple_undirected_graph_wire_bytes(graph) == len(
        encode_strict_json(graph.model_dump(mode="json"))
    )


def test_indexed_null_graph_is_a_canonical_value() -> None:
    """The shared indexed value represents the valid null graph."""

    graph = IndexedSimpleUndirectedGraph(vertex_count=0, edges=())

    assert graph.vertex_count == 0
    assert IndexedSimpleUndirectedGraph.model_validate(graph.model_dump()) == graph
    assert (
        IndexedSimpleUndirectedGraph.model_validate_json(graph.model_dump_json())
        == graph
    )


def test_indexed_null_graph_admits_no_edge() -> None:
    """With zero vertices every edge endpoint lies outside 0..n-1."""

    with pytest.raises(ValidationError):
        IndexedSimpleUndirectedGraph(vertex_count=0, edges=((0, 1),))


def test_indexed_graph_rejects_negative_vertex_counts() -> None:
    with pytest.raises(ValidationError):
        IndexedSimpleUndirectedGraph(vertex_count=-1, edges=())


def test_indexed_graph_requires_canonical_edge_orientation() -> None:
    """A reversed pair must be rejected, not silently retained: the shared
    value composes on serialized equality, so ``(1, 0)`` and ``(0, 1)``
    cannot both represent the same undirected edge."""

    with pytest.raises(ValidationError, match="left < right"):
        IndexedSimpleUndirectedGraph(vertex_count=2, edges=((1, 0),))


def test_indexed_graph_rejects_duplicate_edges_across_orientation() -> None:
    """``(0, 1)`` and ``(1, 0)`` are the same undirected edge; submitting
    both can never validate.  As in ``SimpleUndirectedGraph``, the
    per-edge canonical-order rule fires before duplicate detection."""

    with pytest.raises(ValidationError):
        IndexedSimpleUndirectedGraph(vertex_count=2, edges=((1, 0), (0, 1)))
    with pytest.raises(ValidationError, match="left < right"):
        IndexedSimpleUndirectedGraph(vertex_count=3, edges=((0, 1), (2, 0)))


def test_indexed_canonical_edges_round_trip_serialization() -> None:
    """Canonical nonempty values keep their exact edge order through the
    serialized composition boundary."""

    graph = IndexedSimpleUndirectedGraph(vertex_count=3, edges=((0, 1), (1, 2), (0, 2)))

    assert graph.edges == ((0, 1), (1, 2), (0, 2))
    assert IndexedSimpleUndirectedGraph.model_validate(graph.model_dump()) == graph
    assert (
        IndexedSimpleUndirectedGraph.model_validate_json(graph.model_dump_json())
        == graph
    )
