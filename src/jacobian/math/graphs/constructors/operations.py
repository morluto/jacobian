"""Exact native graph constructor operations."""

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
from jacobian.math.graphs.values import (
    IndexedSimpleUndirectedGraph,
    SimpleUndirectedGraph,
)


def _run_hypercube_graph(request: HypercubeGraphRequest) -> HypercubeGraphResult:
    """Construct the d-dimensional hypercube graph Q_d."""
    dimension = request.dimension
    vertex_count = 1 << dimension
    edges: list[tuple[int, int]] = []
    for vertex in range(vertex_count):
        for bit in range(dimension):
            neighbor = vertex ^ (1 << bit)
            if neighbor > vertex:
                edges.append((vertex, neighbor))
    return HypercubeGraphResult(
        dimension=dimension,
        graph=IndexedSimpleUndirectedGraph(
            vertex_count=vertex_count,
            edges=tuple(edges),
        ),
    )


def _to_word(value: int, dimension: int) -> tuple[int, ...]:
    """Convert an integer to a dimension-digit base-4 word."""
    word = []
    for _ in range(dimension):
        word.append(value % 4)
        value //= 4
    return tuple(reversed(word))


def _keller_adjacent(left: tuple[int, ...], right: tuple[int, ...]) -> bool:
    """Return whether two words satisfy Keller adjacency."""
    has_diff_2_mod_4 = False
    hamming = 0
    for left_value, right_value in zip(left, right, strict=True):
        if left_value != right_value:
            hamming += 1
            if abs(left_value - right_value) == 2:
                has_diff_2_mod_4 = True
    return has_diff_2_mod_4 and hamming >= 2


def _run_keller_graph(request: KellerGraphRequest) -> KellerGraphResult:
    """Construct the Keller graph K_d."""
    dimension = request.dimension
    if dimension == 0:
        return KellerGraphResult(
            dimension=dimension,
            graph=IndexedSimpleUndirectedGraph(vertex_count=1, edges=()),
        )
    vertex_count = 4**dimension
    edges: list[tuple[int, int]] = []
    for left in range(vertex_count):
        left_word = _to_word(left, dimension)
        for right in range(left + 1, vertex_count):
            if _keller_adjacent(left_word, _to_word(right, dimension)):
                edges.append((left, right))
    return KellerGraphResult(
        dimension=dimension,
        graph=IndexedSimpleUndirectedGraph(
            vertex_count=vertex_count,
            edges=tuple(edges),
        ),
    )


def _run_triangle_profile(request: TriangleProfileRequest) -> TriangleProfileResult:
    """Compute the complete triangle profile of a finite simple graph."""
    graph = request.graph
    admission = admit_triangle_profile(graph)
    triangles = tuple(
        TriangleProfileRow(
            vertices=(
                graph.vertices[left],
                graph.vertices[middle],
                graph.vertices[right],
            )
        )
        for left, middle, right in admission.triangle_indices
    )
    return TriangleProfileResult(
        source=graph,
        triangles=triangles,
        triangle_count=admission.triangle_count,
    )


def construct_hypercube_graph(dimension: int) -> HypercubeGraphResult:
    """Construct ``Q_dimension`` from its mathematical dimension."""
    return _run_hypercube_graph(HypercubeGraphRequest(dimension=dimension))


def construct_keller_graph(dimension: int) -> KellerGraphResult:
    """Construct the Keller graph of the requested dimension."""
    return _run_keller_graph(KellerGraphRequest(dimension=dimension))


def compute_triangle_profile(graph: SimpleUndirectedGraph) -> TriangleProfileResult:
    """Compute the complete triangle profile of one canonical graph."""
    if not isinstance(graph, SimpleUndirectedGraph):
        raise TypeError("compute_triangle_profile expects a SimpleUndirectedGraph")
    return _run_triangle_profile(TriangleProfileRequest(graph=graph))


__all__ = [
    "compute_triangle_profile",
    "construct_hypercube_graph",
    "construct_keller_graph",
]
