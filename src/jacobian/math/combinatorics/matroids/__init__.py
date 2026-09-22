"""Native APIs for linear matroid operations."""

from jacobian.math.combinatorics.matroids._models import LinearMatroid
from jacobian.math.combinatorics.matroids.intersection import matroid_intersection
from jacobian.math.combinatorics.matroids.operations import (
    matroid_closure,
    matroid_rank,
    maximum_weight_basis_result,
    verify_closure,
    verify_maximum_weight_basis,
)

__all__ = [
    "LinearMatroid",
    "matroid_closure",
    "matroid_intersection",
    "matroid_rank",
    "maximum_weight_basis_result",
    "verify_closure",
    "verify_maximum_weight_basis",
]
