"""Exact graph constructor kernels: hypercube, Keller, and triangle profiles."""

from __future__ import annotations

from jacobian.math.graphs.constructors._bounds import admit_triangle_profile
from jacobian.math.graphs.constructors._models import (
    HypercubeGraphRequest,
    HypercubeGraphResult,
    KellerGraphRequest,
    KellerGraphResult,
    TriangleProfileRequest,
    TriangleProfileResult,
    TriangleProfileRow,
)
from jacobian.math.graphs.values import IndexedSimpleUndirectedGraph


def _run_hypercube_graph(request: HypercubeGraphRequest) -> HypercubeGraphResult:
    """Construct the d-dimensional hypercube graph Q_d.

    Vertices are indexed 0..2^d-1. Two vertices are adjacent iff they
    differ in exactly one bit.
    """
    d = request.dimension
    n = 1 << d  # 2^d vertices

    edges: list[tuple[int, int]] = []

    for v in range(n):
        for bit in range(d):
            neighbor = v ^ (1 << bit)
            if neighbor > v:
                edges.append((v, neighbor))

    return HypercubeGraphResult(
        dimension=d,
        graph=IndexedSimpleUndirectedGraph(
            vertex_count=n,
            edges=tuple(edges),
        ),
    )


def _run_keller_graph(request: KellerGraphRequest) -> KellerGraphResult:
    """Construct the Keller graph K_d.

    Vertices are words in {0,1,2,3}^d indexed 0..4^d-1 in lexicographic
    base-4 word order. Two distinct words u, v are adjacent iff they
    differ by 2 (mod 4) in at least one coordinate AND differ in at
    least two coordinates overall (Hamming distance >= 2).
    """
    d = request.dimension

    if d == 0:
        # K_0: one vertex, no edges
        return KellerGraphResult(
            dimension=0,
            graph=IndexedSimpleUndirectedGraph(
                vertex_count=1,
                edges=(),
            ),
        )

    n = 4**d
    edges: list[tuple[int, int]] = []

    for i in range(n):
        wi = _to_word(i, d)
        for j in range(i + 1, n):
            wj = _to_word(j, d)
            if _keller_adjacent(wi, wj):
                edges.append((i, j))

    return KellerGraphResult(
        dimension=d,
        graph=IndexedSimpleUndirectedGraph(
            vertex_count=n,
            edges=tuple(edges),
        ),
    )


def _to_word(n: int, d: int) -> tuple[int, ...]:
    """Convert integer n to a d-digit base-4 word."""
    word = []
    for _ in range(d):
        word.append(n % 4)
        n //= 4
    return tuple(reversed(word))


def _keller_adjacent(wi: tuple[int, ...], wj: tuple[int, ...]) -> bool:
    """Keller adjacency: differ by 2 (mod 4) in some coord AND Hamming >= 2."""
    has_diff_2_mod_4 = False
    hamming = 0
    for a, b in zip(wi, wj, strict=True):
        if a != b:
            hamming += 1
            if abs(a - b) == 2:
                has_diff_2_mod_4 = True
    return has_diff_2_mod_4 and hamming >= 2


def _run_triangle_profile(request: TriangleProfileRequest) -> TriangleProfileResult:
    """Compute the complete triangle profile of a finite simple undirected graph.

    For every unordered triple of vertices, check whether all three edges
    exist in the source graph. The result is a complete, source-bound list
    of triangles.
    """
    graph = request.graph
    vertex_list = graph.vertices
    admission = admit_triangle_profile(graph)
    triangles = tuple(
        TriangleProfileRow(
            vertices=(vertex_list[left], vertex_list[middle], vertex_list[right])
        )
        for left, middle, right in admission.triangle_indices
    )

    return TriangleProfileResult(
        source=graph,
        triangles=triangles,
        triangle_count=admission.triangle_count,
    )


__all__ = [
    "_run_hypercube_graph",
    "_run_keller_graph",
    "_run_triangle_profile",
]
