"""Binary code distance graph kernel."""

from __future__ import annotations

from jacobian.math.combinatorics.codes.nonlinear.distance_graph._models import (
    BinaryCodeDistanceGraphResult,
)
from jacobian.math.combinatorics.codes.nonlinear.values import ExplicitBinaryCode
from jacobian.math.graphs.values import IndexedSimpleUndirectedGraph

__all__ = ["compute_distance_graph"]


def compute_distance_graph(
    source: ExplicitBinaryCode,
    target_distance: int,
) -> BinaryCodeDistanceGraphResult:
    """Construct the graph whose edges join codeword pairs at a given Hamming distance.

    Vertices are the canonical codeword indices (0..M-1). An edge (i,j) exists
    iff the Hamming distance between codewords[i] and codewords[j] equals
    target_distance.
    """
    codewords = source.codewords
    n = len(codewords)
    edges: list[tuple[int, int]] = []

    for i in range(n):
        for j in range(i + 1, n):
            dist = sum(
                1 for a, b in zip(codewords[i], codewords[j], strict=True) if a != b
            )
            if dist == target_distance:
                edges.append((i, j))

    graph = IndexedSimpleUndirectedGraph(
        vertex_count=n,
        edges=tuple(edges),
    )
    return BinaryCodeDistanceGraphResult(
        source=source,
        target_distance=target_distance,
        graph=graph,
        edge_count=len(edges),
    )
