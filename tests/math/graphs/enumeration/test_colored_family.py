"""Exhaustive labeled families and a composed chordality control."""

from collections import Counter
from itertools import combinations, permutations, product
from typing import Any

import networkx as nx

from jacobian.math.graphs.enumeration import (
    ColorPairCost,
    ConnectedColoredGraphFamily,
    connected_colored_graphs,
)
from jacobian.math.graphs.enumeration._models import ConnectedColoredGraphsRequest


def _native(request: ConnectedColoredGraphsRequest) -> ConnectedColoredGraphFamily:
    return connected_colored_graphs(
        request.palette,
        request.edge_costs,
        vertex_bound=request.vertex_bound,
        cost_bound=request.cost_bound,
    )


def _request(vertices: int, cost: int) -> ConnectedColoredGraphsRequest:
    return ConnectedColoredGraphsRequest(
        palette=("outside", "root"),
        edge_costs=(
            ColorPairCost(colors=("outside", "outside"), cost=2),
            ColorPairCost(colors=("outside", "root"), cost=1),
        ),
        vertex_bound=vertices,
        cost_bound=cost,
    )


def _key(
    colors: tuple[str, ...], edges: tuple[tuple[int, int], ...]
) -> tuple[Any, ...]:
    return min(
        (
            tuple(colors[i] for i in order),
            tuple(
                sorted(
                    tuple(sorted((order.index(a), order.index(b)))) for a, b in edges
                )
            ),
        )
        for order in permutations(range(len(colors)))
    )


def test_small_family_equals_exhaustive_labeled_generation() -> None:
    expected = set()
    for n in range(2, 5):
        pairs = tuple(combinations(range(n), 2))
        for flags in product((False, True), repeat=len(pairs)):
            edges = tuple(e for e, flag in zip(pairs, flags, strict=True) if flag)
            graph: nx.Graph[Any] = nx.Graph()
            graph.add_nodes_from(range(n))
            graph.add_edges_from(edges)
            if not nx.is_connected(graph):
                continue
            for colors in product(("outside", "root"), repeat=n):
                if any(colors[a] == colors[b] == "root" for a, b in edges):
                    continue
                cost = sum(2 if colors[a] == colors[b] else 1 for a, b in edges)
                if cost <= 3:
                    expected.add(_key(colors, edges))
    result = _native(_request(4, 3))
    actual = {
        _key(
            g.vertex_colors,
            tuple(
                (g.graph.vertices.index(a), g.graph.vertices.index(b))
                for a, b in g.graph.edges
            ),
        )
        for g in result.graphs
    }
    assert actual == expected and len(actual) == len(result.graphs)


def test_cost_six_family_and_independent_chordality_composition() -> None:
    result = _native(_request(7, 6))
    assert len(result.graphs) == 111
    counts: Counter[int] = Counter()
    for colored in result.graphs:
        labels = colored.graph.vertices
        colors = dict(zip(labels, colored.vertex_colors, strict=True))
        edges = set(colored.graph.edges)
        cost = sum(2 if colors[a] == colors[b] else 1 for a, b in edges)
        graph: nx.Graph[Any] = nx.Graph()
        graph.add_nodes_from(labels)
        for a, b in combinations(labels, 2):
            same = colors[a] == colors[b]
            present = colors[a] == "root" if same else (a, b) not in edges
            if same and colors[a] == "outside":
                present = (a, b) in edges
            if present:
                graph.add_edge(a, b)
        if nx.is_chordal(graph):
            counts[cost] += 1
    assert [counts[i] for i in range(1, 7)] == [1, 3, 4, 11, 19, 57]


def test_single_color_and_forbidden_edges() -> None:
    request = ConnectedColoredGraphsRequest(
        palette=("red",),
        edge_costs=(ColorPairCost(colors=("red", "red"), cost=1),),
        vertex_bound=3,
        cost_bound=3,
    )
    result = _native(request)
    assert [len(g.graph.edges) for g in result.graphs] == [1, 2, 3]
    assert all(set(g.vertex_colors) == {"red"} for g in result.graphs)
    assert not _native(request.model_copy(update={"edge_costs": ()})).graphs
    assert not _native(request.model_copy(update={"cost_bound": 0})).graphs


def test_excessive_coloring_family_is_rejected() -> None:
    import pytest

    from jacobian.catalog.models import OperationResourceAdmissionError

    palette = tuple(str(i) for i in range(8))
    request = ConnectedColoredGraphsRequest(
        palette=palette,
        edge_costs=tuple(
            ColorPairCost(colors=(a, b), cost=1)
            for a in palette
            for b in palette
            if a <= b
        ),
        vertex_bound=7,
        cost_bound=21,
    )
    with pytest.raises(OperationResourceAdmissionError, match="search units"):
        _native(request)


def test_native_invalid_bounds_fail_before_enumeration() -> None:
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="less than or equal to 7"):
        connected_colored_graphs(("red",), (), vertex_bound=8, cost_bound=1)
    with pytest.raises(ValidationError, match="distinct and increasing"):
        connected_colored_graphs(("red", "red"), (), vertex_bound=2, cost_bound=1)
