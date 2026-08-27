"""Graph constructor operations."""

from jacobian.math.graphs.constructors._models import (
    HypercubeGraphRequest,
    HypercubeGraphResult,
    KellerGraphRequest,
    KellerGraphResult,
    TriangleProfileRequest,
    TriangleProfileResult,
)
from jacobian.math.graphs.constructors._operations import (
    compute_triangle_profile,
    construct_hypercube_graph,
    construct_keller_graph,
)

__all__ = [
    "HypercubeGraphRequest",
    "HypercubeGraphResult",
    "KellerGraphRequest",
    "KellerGraphResult",
    "TriangleProfileRequest",
    "TriangleProfileResult",
    "compute_triangle_profile",
    "construct_hypercube_graph",
    "construct_keller_graph",
]
