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
    construct_hypercube_graph,
    construct_keller_graph,
    compute_triangle_profile,
)

__all__ = [
    "HypercubeGraphRequest",
    "HypercubeGraphResult",
    "KellerGraphRequest",
    "KellerGraphResult",
    "TriangleProfileRequest",
    "TriangleProfileResult",
    "construct_hypercube_graph",
    "construct_keller_graph",
    "compute_triangle_profile",
]
