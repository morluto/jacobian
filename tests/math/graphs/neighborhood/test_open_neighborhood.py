from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.graphs.neighborhood._models import (
    NeighborhoodRequest,
)
from jacobian.math.graphs.neighborhood._tools import compute_open_neighborhood
from jacobian.math.graphs.neighborhood.operations import open_neighborhood
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _graph(vertices, edges):
    return SimpleUndirectedGraph(
        vertices=tuple(vertices),
        edges=tuple((a, b) for a, b in edges),
    )


def test_empty_selected_set() -> None:
    """Empty S has empty neighbourhood."""
    g = _graph(["a", "b"], [("a", "b")])
    result = open_neighborhood(g, ())
    assert result.neighborhood == ()


def test_single_vertex_neighborhood() -> None:
    """N({a}) in path a-b-c is {b}."""
    g = _graph(["a", "b", "c"], [("a", "b"), ("b", "c")])
    result = open_neighborhood(g, ("a",))
    assert result.neighborhood == ("b",)


def test_selected_vertex_excluded_from_neighborhood() -> None:
    """Open neighbourhood excludes selected vertices even if adjacent."""
    g = _graph(["a", "b", "c"], [("a", "b"), ("b", "c")])
    result = open_neighborhood(g, ("a", "b"))
    assert result.neighborhood == ("c",)


def test_edgeless_graph() -> None:
    """Edgeless graph has empty neighbourhood for any S."""
    g = _graph(["a", "b", "c"], ())
    result = open_neighborhood(g, ("a", "b"))
    assert result.neighborhood == ()


def test_all_vertices_selected() -> None:
    """All vertices selected means empty neighbourhood."""
    g = _graph(["a", "b", "c"], [("a", "b"), ("b", "c")])
    result = open_neighborhood(g, ("a", "b", "c"))
    assert result.neighborhood == ()


def test_overlapping_neighbours() -> None:
    """Two selected vertices with overlapping neighbours produce a union."""
    g = _graph(
        ["a", "b", "c", "d"],
        [("a", "c"), ("b", "c"), ("a", "d"), ("b", "d")],
    )
    result = open_neighborhood(g, ("a", "b"))
    assert set(result.neighborhood) == {"c", "d"}


def test_star_graph() -> None:
    """Star with center selected: all leaves are in the neighbourhood."""
    g = _graph(["c", "l1", "l2", "l3"], [("c", "l1"), ("c", "l2"), ("c", "l3")])
    result = open_neighborhood(g, ("c",))
    assert set(result.neighborhood) == {"l1", "l2", "l3"}


def test_star_graph_leaves_selected() -> None:
    """Star with leaves selected: center is in the neighbourhood."""
    g = _graph(["c", "l1", "l2", "l3"], [("c", "l1"), ("c", "l2"), ("c", "l3")])
    result = open_neighborhood(g, ("l1", "l2", "l3"))
    assert result.neighborhood == ("c",)


def test_canonical_vertex_order() -> None:
    """Neighbourhood preserves source-vertex order."""
    g = _graph(["z", "a", "m"], [("a", "z"), ("m", "z")])
    result = open_neighborhood(g, ("z",))
    assert result.neighborhood == ("a", "m")


def test_result_retains_source() -> None:
    """Result retains the original graph and selected vertices."""
    g = _graph(["a", "b"], [("a", "b")])
    result = open_neighborhood(g, ("a",))
    assert result.graph == g
    assert result.selected_vertices == ("a",)


def test_request_rejects_nonexistent_vertex() -> None:
    g = _graph(["a", "b"], [("a", "b")])
    with pytest.raises(ValidationError):
        NeighborhoodRequest(graph=g, selected_vertices=("c",))


def test_request_rejects_duplicate_selected() -> None:
    g = _graph(["a", "b"], [("a", "b")])
    with pytest.raises(ValidationError):
        NeighborhoodRequest(graph=g, selected_vertices=("a", "a"))


def test_catalog_operation_runs() -> None:
    """The _tools adapter produces the same result."""
    g = _graph(["a", "b", "c"], [("a", "b"), ("b", "c")])
    request = NeighborhoodRequest(graph=g, selected_vertices=("a",))
    result = compute_open_neighborhood(request)
    assert result.neighborhood == ("b",)


def test_disconnected_graph() -> None:
    """Disconnected graph: selected vertex in one component doesn't affect the other."""
    g = _graph(["a", "b", "c", "d"], [("a", "b"), ("c", "d")])
    result = open_neighborhood(g, ("a",))
    assert result.neighborhood == ("b",)


def test_complete_graph_neighborhood() -> None:
    """K4: N({a}) = {b,c,d}."""
    g = _graph(
        ["a", "b", "c", "d"],
        [("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"), ("b", "d"), ("c", "d")],
    )
    result = open_neighborhood(g, ("a",))
    assert set(result.neighborhood) == {"b", "c", "d"}
