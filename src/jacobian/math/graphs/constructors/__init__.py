"""Native mathematical wrappers for graph constructor operations."""

from jacobian.math.graphs.constructors._models import (
    HypercubeGraphRequest,
    HypercubeGraphResult,
    KellerGraphRequest,
    KellerGraphResult,
    TriangleProfileRequest,
    TriangleProfileResult,
    TriangleProfileRow,
)
from jacobian.math.graphs.constructors._operations import (
    _run_hypercube_graph,
    _run_keller_graph,
    _run_triangle_profile,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


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
    "HypercubeGraphResult",
    "KellerGraphResult",
    "TriangleProfileResult",
    "TriangleProfileRow",
    "compute_triangle_profile",
    "construct_hypercube_graph",
    "construct_keller_graph",
]
