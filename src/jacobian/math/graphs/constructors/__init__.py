"""Native mathematical wrappers for graph constructor operations."""

from jacobian.math.graphs.constructors._models import (
    HypercubeGraphResult,
    KellerGraphResult,
    TriangleProfileResult,
    TriangleProfileRow,
)
from jacobian.math.graphs.constructors.operations import (
    compute_triangle_profile,
    construct_hypercube_graph,
    construct_keller_graph,
    verify_hypercube_graph,
    verify_keller_graph,
    verify_triangle_profile,
)

__all__ = [
    "HypercubeGraphResult",
    "KellerGraphResult",
    "TriangleProfileResult",
    "TriangleProfileRow",
    "compute_triangle_profile",
    "construct_hypercube_graph",
    "construct_keller_graph",
    "verify_hypercube_graph",
    "verify_keller_graph",
    "verify_triangle_profile",
]
