"""Edge case tests for the SimpleUndirectedGraph validator."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.canonical import encode_strict_json
from jacobian.math.graphs.values import (
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
