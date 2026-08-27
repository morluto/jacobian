"""Exact 3-term progression hypergraph kernel for finite cyclic groups."""

from __future__ import annotations

from jacobian.math.combinatorics._progression_hypergraph_models import (
    ProgressionHypergraphRequest,
    ProgressionHypergraphResult,
)
from jacobian.math.hypergraphs._models import FiniteHypergraph


def construct_3term_progression_hypergraph(
    request: ProgressionHypergraphRequest,
) -> ProgressionHypergraphResult:
    """Construct the 3-uniform hypergraph of 3-APs in Z/nZ.

    Vertices: 0, 1, ..., n-1.
    Edges: all {a, a+d, a+2d} for d = 1, ..., n-1 (mod n), with a in Z/nZ.
    Each edge is a set of 3 distinct elements forming an arithmetic progression.
    """
    n = request.group_order
    vertices = tuple(str(i) for i in range(n))

    edges_set: set[frozenset[int]] = set()
    for d in range(1, n):
        for a in range(n):
            v0 = a
            v1 = (a + d) % n
            v2 = (a + 2 * d) % n
            if len({v0, v1, v2}) == 3:
                edges_set.add(frozenset({v0, v1, v2}))

    sorted_edges = sorted(edges_set, key=lambda s: tuple(sorted(s)))
    edges = tuple(
        (f"e{i}", tuple(sorted(str(v) for v in edge)))
        for i, edge in enumerate(sorted_edges)
    )

    return ProgressionHypergraphResult(
        group_order=n,
        hypergraph=FiniteHypergraph(vertices=vertices, edges=edges),
    )


__all__ = ["construct_3term_progression_hypergraph"]
