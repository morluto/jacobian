"""Edge case tests for the SimpleUndirectedGraph validator."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.graphs.values import SimpleUndirectedGraph


def test_graph_rejects_non_nfc_vertices() -> None:
    with pytest.raises(ValidationError, match="NFC"):
        SimpleUndirectedGraph(vertices=("e\u0301",), edges=())


def test_graph_rejects_duplicate_vertices() -> None:
    with pytest.raises(ValidationError, match="unique"):
        SimpleUndirectedGraph(vertices=("a", "a"), edges=())


def test_graph_rejects_edges_with_undeclared_vertices() -> None:
    with pytest.raises(ValidationError, match="declared vertices"):
        SimpleUndirectedGraph(vertices=("a",), edges=(("a", "b"),))


def test_graph_rejects_edges_out_of_order() -> None:
    with pytest.raises(ValidationError, match="declared vertices"):
        SimpleUndirectedGraph(vertices=("a", "b"), edges=(("b", "a"),))


def test_graph_rejects_duplicate_edges() -> None:
    with pytest.raises(ValidationError, match="unique"):
        SimpleUndirectedGraph(
            vertices=("a", "b"),
            edges=(("a", "b"), ("a", "b")),
        )
