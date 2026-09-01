from __future__ import annotations

from itertools import combinations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.monochromatic_clique._models import (
    MonochromaticCliqueHypergraphRequest,
)
from jacobian.math.graphs.monochromatic_clique.operations import (
    construct_monochromatic_clique_hypergraph,
)
from jacobian.math.graphs.values import ColoredUndirectedGraph, SimpleUndirectedGraph


def _k4_red() -> ColoredUndirectedGraph:
    return ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(
            vertices=("0", "1", "2", "3"),
            edges=(
                ("0", "1"),
                ("0", "2"),
                ("0", "3"),
                ("1", "2"),
                ("1", "3"),
                ("2", "3"),
            ),
        ),
        edge_colors=("red", "red", "red", "red", "red", "red"),
    )


def _k4_mixed() -> ColoredUndirectedGraph:
    return ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(
            vertices=("0", "1", "2", "3"),
            edges=(
                ("0", "1"),
                ("0", "2"),
                ("0", "3"),
                ("1", "2"),
                ("1", "3"),
                ("2", "3"),
            ),
        ),
        edge_colors=("red", "red", "red", "blue", "blue", "blue"),
    )


def test_all_red_k4_t3() -> None:
    """All-red K4 with t=3: four monochromatic 3-cliques."""
    result = construct_monochromatic_clique_hypergraph(_k4_red(), 3)
    assert len(result.hypergraph.edges) == 4


def test_mixed_k4_t3() -> None:
    """Mixed K4 with edges 0-1-2 red, 1-2-3 blue: check clique counts."""
    result = construct_monochromatic_clique_hypergraph(_k4_mixed(), 3)
    # Only {0,1,2} is a monochromatic triangle (all red)
    edges = [frozenset(members) for _, members in result.hypergraph.edges]
    assert frozenset({"1", "2", "3"}) in edges
    assert len(edges) == 1


def test_t2() -> None:
    """t=2: every edge is a monochromatic 2-clique."""
    result = construct_monochromatic_clique_hypergraph(_k4_red(), 2)
    assert len(result.hypergraph.edges) == 6  # C(4,2) = 6


def test_all_red_k9_is_admitted_when_only_one_clique_is_materialized() -> None:
    vertices = tuple(str(index) for index in range(9))
    graph = SimpleUndirectedGraph(
        vertices=vertices,
        edges=tuple(combinations(vertices, 2)),
    )
    result = construct_monochromatic_clique_hypergraph(
        ColoredUndirectedGraph(graph=graph, edge_colors=("red",) * len(graph.edges)),
        9,
    )
    assert len(result.hypergraph.edges) == 1


def test_edgeless_graph_accepts_empty_edge_coloring() -> None:
    graph = SimpleUndirectedGraph(vertices=("0",), edges=())
    result = construct_monochromatic_clique_hypergraph(
        ColoredUndirectedGraph(graph=graph), 2
    )
    assert result.hypergraph.edges == ()


def test_replay_monochromatic() -> None:
    """Every hyperedge induces a monochromatic clique."""
    result = construct_monochromatic_clique_hypergraph(_k4_mixed(), 3)
    graph = _k4_mixed()
    edge_colors = {}
    for (a, b), c in zip(graph.graph.edges, graph.edge_colors, strict=True):
        edge_colors[(a, b)] = c
    for _, members in result.hypergraph.edges:
        colors = set()
        for a, b in combinations(members, 2):
            colors.add(edge_colors.get((a, b), edge_colors.get((b, a))))
        assert len(colors) == 1


def test_exhaustive_comparison() -> None:
    """Compare against independent enumeration."""
    result = construct_monochromatic_clique_hypergraph(_k4_red(), 3)
    graph = _k4_red()
    edge_colors = {}
    for (a, b), c in zip(graph.graph.edges, graph.edge_colors, strict=True):
        edge_colors[(a, b)] = c
    expected = set()
    for subset in combinations(["0", "1", "2", "3"], 3):
        colors = set()
        is_clique = True
        for a, b in combinations(subset, 2):
            color = edge_colors.get((a, b), edge_colors.get((b, a)))
            if color is None:
                is_clique = False
                break
            colors.add(color)
        if is_clique and len(colors) == 1:
            expected.add(frozenset(subset))
    actual = {frozenset(members) for _, members in result.hypergraph.edges}
    assert actual == expected


def test_vertex_preservation() -> None:
    """Hypergraph preserves all source vertices."""
    result = construct_monochromatic_clique_hypergraph(_k4_red(), 3)
    assert set(result.hypergraph.vertices) == {"0", "1", "2", "3"}


def test_result_preserves_source() -> None:
    """Result retains the source graph and clique order."""
    cg = _k4_red()
    result = construct_monochromatic_clique_hypergraph(cg, 2)
    assert result.colored_graph == cg
    assert result.clique_order == 2


def test_native_admission_rejects_hypergraph_edge_and_incidence_overflow() -> None:
    vertices = tuple(f"{index:02}" for index in range(20))
    graph = SimpleUndirectedGraph(
        vertices=vertices,
        edges=tuple(
            (vertices[left], vertices[right])
            for left in range(20)
            for right in range(left + 1, 20)
        ),
    )
    colored_graph = ColoredUndirectedGraph(
        graph=graph, edge_colors=("red",) * len(graph.edges)
    )
    with pytest.raises(OperationDomainValidationError, match="hypergraph bound"):
        construct_monochromatic_clique_hypergraph(colored_graph, 5)


def test_native_admission_rejects_missing_total_coloring() -> None:
    with pytest.raises(OperationDomainValidationError, match="edge_colors"):
        construct_monochromatic_clique_hypergraph(
            _k4_red().model_copy(update={"edge_colors": ()}), 3
        )


def test_native_admission_rejects_labels_outside_the_hypergraph_carrier() -> None:
    vertices = tuple("x" * 400_000 + str(index) for index in range(4))
    graph = SimpleUndirectedGraph(
        vertices=vertices,
        edges=tuple(
            (vertices[left], vertices[right])
            for left in range(4)
            for right in range(left + 1, 4)
        ),
    )
    colored_graph = ColoredUndirectedGraph.model_construct(
        graph=graph, edge_colors=("red",) * len(graph.edges)
    )
    with pytest.raises(OperationDomainValidationError, match="label bound"):
        construct_monochromatic_clique_hypergraph(colored_graph, 2)


def test_request_keeps_structural_validation_separate_from_domain_admission() -> None:
    request = MonochromaticCliqueHypergraphRequest(
        colored_graph=_k4_red(), clique_order=3
    )
    assert request.clique_order == 3
