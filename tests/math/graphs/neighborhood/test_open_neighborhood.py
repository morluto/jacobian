from __future__ import annotations

from collections.abc import Sequence

import pytest

from jacobian.canonical import CanonicalLimits, encode_strict_json
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.neighborhood._models import (
    NeighborhoodRequest,
)
from jacobian.math.graphs.neighborhood._tools import compute_open_neighborhood
from jacobian.math.graphs.neighborhood.operations import open_neighborhood
from jacobian.math.graphs.values import (
    MAX_INDEXED_SIMPLE_GRAPH_VERTICES,
    SimpleUndirectedGraph,
)


def _graph(
    vertices: Sequence[str], edges: Sequence[tuple[str, str]]
) -> SimpleUndirectedGraph:
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


def test_issue_path_fixture_has_the_exact_open_neighborhood() -> None:
    g = _graph(["0", "1", "2", "3"], [("0", "1"), ("1", "2"), ("2", "3")])
    result = open_neighborhood(g, ("1",))
    assert result.neighborhood == ("0", "2")


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


def test_native_operation_rejects_nonexistent_vertex() -> None:
    g = _graph(["a", "b"], [("a", "b")])
    with pytest.raises(
        OperationDomainValidationError,
        match="every selected vertex must be a declared graph vertex",
    ):
        open_neighborhood(g, ("c",))


def test_native_operation_rejects_an_oversized_raw_selection_before_hashing() -> None:
    g = _graph(["a"], ())
    with pytest.raises(
        OperationDomainValidationError,
        match=(f"at most {MAX_INDEXED_SIMPLE_GRAPH_VERTICES} raw vertices"),
    ):
        open_neighborhood(g, ("a",) * (MAX_INDEXED_SIMPLE_GRAPH_VERTICES + 1))


def test_catalog_request_rejects_an_oversized_raw_selection() -> None:
    g = _graph(["a"], ())
    with pytest.raises(ValueError, match="raw tuple-length bound"):
        NeighborhoodRequest(
            graph=g,
            selected_vertices=("a",) * (MAX_INDEXED_SIMPLE_GRAPH_VERTICES + 1),
        )


def test_catalog_request_reuses_selection_admission() -> None:
    g = _graph(["a", "b"], [("a", "b")])
    request = NeighborhoodRequest(graph=g, selected_vertices=("c",))
    with pytest.raises(OperationDomainValidationError, match="every selected vertex"):
        compute_open_neighborhood(request)


def test_selected_vertices_are_normalized_as_a_set_in_source_order() -> None:
    g = _graph(["a", "b", "c"], [("a", "c"), ("b", "c")])
    result = open_neighborhood(g, ("b", "a", "b"))
    assert result.selected_vertices == ("a", "b")
    assert result.neighborhood == ("c",)


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


def test_every_neighborhood_vertex_replays_against_an_incident_edge() -> None:
    g = _graph(
        ["a", "b", "c", "d"],
        [("a", "c"), ("b", "c"), ("b", "d")],
    )
    result = open_neighborhood(g, ("a", "b"))
    selected = set(result.selected_vertices)
    edges = {frozenset(edge) for edge in result.graph.edges}
    assert all(
        any(frozenset((source, neighbor)) in edges for source in selected)
        for neighbor in result.neighborhood
    )


def test_native_result_does_not_inherit_canonical_output_budget() -> None:
    long_label = "z" * 4_000_000
    g = _graph(["a", long_label], [("a", long_label)])
    request = NeighborhoodRequest(graph=g, selected_vertices=("a",))
    assert (
        len(encode_strict_json(request.model_dump(mode="json")))
        <= CanonicalLimits().max_input_bytes
    )

    native_result = open_neighborhood(g, ("a",))
    assert native_result.neighborhood == (long_label,)

    assert compute_open_neighborhood(request) == native_result
