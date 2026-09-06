"""Exact native graph constructor operations."""

from __future__ import annotations

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.constructors._bounds import (
    MAX_TRIANGLE_PROFILE_RETAINED_LABEL_CHARACTERS,
    MAX_TRIANGLE_PROFILE_ROWS,
    admit_hypercube_dimension,
    admit_keller_dimension,
    admit_triangle_profile,
)
from jacobian.math.graphs.constructors._models import (
    HypercubeGraphResult,
    KellerGraphResult,
    TriangleProfileResult,
    TriangleProfileRow,
)
from jacobian.math.graphs.values import (
    IndexedSimpleUndirectedGraph,
    SimpleUndirectedGraph,
)


def construct_hypercube_graph(dimension: int) -> HypercubeGraphResult:
    """Construct the d-dimensional hypercube graph Q_d."""
    admit_hypercube_dimension(dimension)
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


def construct_keller_graph(dimension: int) -> KellerGraphResult:
    """Construct the Keller graph K_d."""
    admit_keller_dimension(dimension)
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


def compute_triangle_profile(graph: SimpleUndirectedGraph) -> TriangleProfileResult:
    """Compute the complete triangle profile of a finite simple graph."""
    if not isinstance(graph, SimpleUndirectedGraph):
        raise TypeError("compute_triangle_profile expects a SimpleUndirectedGraph")
    admission = admit_triangle_profile(graph)
    triangles: list[TriangleProfileRow] = []
    retained = sum(len(vertex) for vertex in graph.vertices) + sum(
        len(left) + len(right) for left, right in graph.edges
    )
    for left, right in graph.edges:
        left_index = admission.vertex_index[left]
        right_index = admission.vertex_index[right]
        first, second = sorted((left_index, right_index))
        for third in admission.adjacency[first] & admission.adjacency[second]:
            if third > second:
                triangles.append(
                    TriangleProfileRow(
                        vertices=(
                            graph.vertices[first],
                            graph.vertices[second],
                            graph.vertices[third],
                        )
                    )
                )
                retained += (
                    len(graph.vertices[first])
                    + len(graph.vertices[second])
                    + len(graph.vertices[third])
                )
                if retained > MAX_TRIANGLE_PROFILE_RETAINED_LABEL_CHARACTERS:
                    raise OperationDomainValidationError(
                        location=("graph",),
                        code="graph.triangle_profile.retained_labels_exceed_bound",
                        message=(
                            "triangle profile exceeds the retained "
                            "label-character bound"
                        ),
                    )
                if len(triangles) > MAX_TRIANGLE_PROFILE_ROWS:
                    raise OperationDomainValidationError(
                        location=("graph",),
                        code="graph.triangle_profile.row_bound",
                        message=(
                            f"triangle profile has {len(triangles):,} rows, exceeding "
                            f"the {MAX_TRIANGLE_PROFILE_ROWS:,}-row materialization "
                            "bound"
                        ),
                    )
    return TriangleProfileResult(
        source=graph,
        triangles=tuple(triangles),
        triangle_count=len(triangles),
    )


def verify_hypercube_graph(claim: HypercubeGraphResult) -> bool:
    """Check a Q_d claim against its retained dimension without rebuilding rows."""
    try:
        admit_hypercube_dimension(claim.dimension)
    except OperationDomainValidationError:
        return False
    vertex_count = 1 << claim.dimension
    if claim.graph.vertex_count != vertex_count:
        return False
    expected = {
        (vertex, vertex ^ (1 << bit))
        if vertex < vertex ^ (1 << bit)
        else (vertex ^ (1 << bit), vertex)
        for vertex in range(vertex_count)
        for bit in range(claim.dimension)
    }
    return set(claim.graph.edges) == expected


def verify_keller_graph(claim: KellerGraphResult) -> bool:
    """Check a K_d claim against its retained dimension."""
    try:
        admit_keller_dimension(claim.dimension)
    except OperationDomainValidationError:
        return False
    if claim.dimension == 0:
        return claim.graph.vertex_count == 1 and claim.graph.edges == ()
    vertex_count = 4**claim.dimension
    if claim.graph.vertex_count != vertex_count:
        return False
    expected: set[tuple[int, int]] = set()
    for left in range(vertex_count):
        left_word = _to_word(left, claim.dimension)
        for right in range(left + 1, vertex_count):
            if _keller_adjacent(left_word, _to_word(right, claim.dimension)):
                expected.add((left, right))
    return set(claim.graph.edges) == expected


def verify_triangle_profile(claim: TriangleProfileResult) -> bool:
    """Check triangle rows and completeness against the retained source graph."""
    try:
        admission = admit_triangle_profile(claim.source)
    except OperationDomainValidationError:
        return False
    edge_set = {frozenset(edge) for edge in claim.source.edges}
    seen: set[frozenset[str]] = set()
    for row in claim.triangles:
        vertices = row.vertices
        if len(set(vertices)) != 3 or any(
            v not in admission.vertex_index for v in vertices
        ):
            return False
        key = frozenset(vertices)
        if key in seen:
            return False
        seen.add(key)
        if (
            frozenset((vertices[0], vertices[1])) not in edge_set
            or frozenset((vertices[0], vertices[2])) not in edge_set
            or frozenset((vertices[1], vertices[2])) not in edge_set
        ):
            return False
    if claim.triangle_count != len(claim.triangles):
        return False
    # Completeness: enumerate once in the verifier and compare as sets.
    complete: set[frozenset[str]] = set()
    source_vertices = claim.source.vertices
    for left, right in claim.source.edges:
        first, second = sorted(
            (admission.vertex_index[left], admission.vertex_index[right])
        )
        for third in admission.adjacency[first] & admission.adjacency[second]:
            if third > second:
                complete.add(
                    frozenset(
                        (
                            source_vertices[first],
                            source_vertices[second],
                            source_vertices[third],
                        )
                    )
                )
    return seen == complete


__all__ = [
    "compute_triangle_profile",
    "construct_hypercube_graph",
    "construct_keller_graph",
    "verify_hypercube_graph",
    "verify_keller_graph",
    "verify_triangle_profile",
]
