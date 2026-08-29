"""Maximal-clique hypergraph constructor using Bron-Kerbosch with pivoting."""

from __future__ import annotations

from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)
from jacobian.math.graphs.maximal_clique_hypergraph._models import (
    MaximalCliqueHypergraphResult,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

__all__ = ["construct_maximal_clique_hypergraph"]


def construct_maximal_clique_hypergraph(
    graph: SimpleUndirectedGraph,
) -> MaximalCliqueHypergraphResult:
    """Construct the maximal-clique hypergraph of a simple graph.

    The hypergraph has the same vertices as the graph. Each hyperedge is
    one inclusion-maximal complete vertex set of cardinality at least two
    (nontrivial clique). Isolated vertices induce no singleton edge.
    """
    vertices = list(graph.vertices)
    adjacency = _build_adjacency(graph)
    cliques = _find_maximal_cliques(vertices, adjacency)
    nontrivial = [c for c in cliques if len(c) >= 2]

    hyper_vertices = tuple(vertices)
    hyper_edges = []
    for i, clique in enumerate(nontrivial):
        edge_id = f"clique_{i}"
        hyper_edges.append((edge_id, tuple(sorted(clique))))

    hypergraph = FiniteHypergraph(
        vertices=hyper_vertices,
        edges=tuple(hyper_edges),
    )
    return MaximalCliqueHypergraphResult(graph=graph, hypergraph=hypergraph)


def _build_adjacency(graph: SimpleUndirectedGraph) -> dict[str, set[str]]:
    adj: dict[str, set[str]] = {v: set() for v in graph.vertices}
    for a, b in graph.edges:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def _find_maximal_cliques(
    vertices: list[str], adjacency: dict[str, set[str]]
) -> list[list[str]]:
    """Find all maximal cliques using Bron-Kerbosch with pivoting."""
    cliques: list[list[str]] = []
    _bron_kerbosch_pivot(set(), set(vertices), set(), adjacency, cliques)
    return cliques


def _bron_kerbosch_pivot(
    r: set[str],
    p: set[str],
    x: set[str],
    adjacency: dict[str, set[str]],
    cliques: list[list[str]],
) -> None:
    if not p and not x:
        cliques.append(sorted(r))
        return
    pivot = next(iter(p | x)) if (p | x) else None
    if pivot is not None:
        for v in sorted(p - adjacency[pivot]):
            _bron_kerbosch_pivot(
                r | {v},
                p & adjacency[v],
                x & adjacency[v],
                adjacency,
                cliques,
            )
            p = p - {v}
            x = x | {v}
