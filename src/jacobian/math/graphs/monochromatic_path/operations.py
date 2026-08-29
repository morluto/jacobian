"""Monochromatic path hypergraph constructor."""

from __future__ import annotations

from itertools import combinations

from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)
from jacobian.math.graphs.monochromatic_path._models import (
    MonochromaticPathResult,
)
from jacobian.math.graphs.values import ColoredUndirectedGraph

__all__ = ["construct_monochromatic_path_hypergraphs"]


def construct_monochromatic_path_hypergraphs(
    graph: ColoredUndirectedGraph,
) -> MonochromaticPathResult:
    """For each colour, return the hypergraph whose edges are vertex sets
    that admit a monochromatic simple path using only edges of that colour.

    Singletons are included (length-0 path convention).
    """
    vertices = list(graph.graph.vertices)
    edges = list(graph.graph.edges)
    edge_colors = list(graph.edge_colors)

    colours = sorted(set(edge_colors))
    if not colours:
        colours = ["uncolored"]

    colour_to_adjacency: dict[str, dict[str, set[str]]] = {
        c: {v: set() for v in vertices} for c in colours
    }
    for (a, b), c in zip(edges, edge_colors, strict=True):
        colour_to_adjacency[c][a].add(b)
        colour_to_adjacency[c][b].add(a)

    result: dict[str, FiniteHypergraph] = {}

    for colour in colours:
        adj = colour_to_adjacency[colour]
        supports: list[tuple[str, ...]] = []

        for size in range(1, len(vertices) + 1):
            for subset in combinations(vertices, size):
                if _has_hamiltonian_path(subset, adj):
                    supports.append(tuple(sorted(subset)))

        hyper_edges: list[tuple[str, tuple[str, ...]]] = []
        for i, support in enumerate(sorted(supports)):
            hyper_edges.append((f"path_{colour}_{i}", support))

        result[colour] = FiniteHypergraph(
            vertices=tuple(vertices),
            edges=tuple(hyper_edges),
        )

    return MonochromaticPathResult(
        graph=graph,
        colour_to_hypergraph=result,
    )


def _has_hamiltonian_path(
    vertices: tuple[str, ...],
    adjacency: dict[str, set[str]],
) -> bool:
    """Check if the subgraph induced on `vertices` has a Hamiltonian path."""
    n = len(vertices)
    if n == 1:
        return True

    v_set = set(vertices)

    for start in vertices:
        visited: set[str] = {start}
        path = [start]
        if _dfs_path(start, path, visited, n, vertices, adjacency, v_set):
            return True
    return False


def _dfs_path(
    current: str,
    path: list[str],
    visited: set[str],
    n: int,
    vertices: tuple[str, ...],
    adjacency: dict[str, set[str]],
    v_set: set[str],
) -> bool:
    if len(path) == n:
        return True
    for neighbor in sorted(adjacency[current]):
        if neighbor in v_set and neighbor not in visited:
            visited.add(neighbor)
            path.append(neighbor)
            if _dfs_path(neighbor, path, visited, n, vertices, adjacency, v_set):
                return True
            path.pop()
            visited.discard(neighbor)
    return False
