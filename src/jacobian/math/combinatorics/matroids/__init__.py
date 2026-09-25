"""Native APIs for linear matroid operations."""

from jacobian.math.combinatorics.matroids._models import (
    LinearMatroid,
    MatroidWeightedIntersectionResult,
    MatroidWeightFunction,
)
from jacobian.math.combinatorics.matroids.graphic import graphic_matroid
from jacobian.math.combinatorics.matroids.intersection import (
    matroid_common_basis,
    matroid_intersection,
    verify_common_basis_result,
    verify_weighted_intersection_result,
    weighted_intersection_certificate,
)
from jacobian.math.combinatorics.matroids.operations import (
    matroid_closure,
    matroid_rank,
    maximum_weight_basis_result,
    maximum_weight_independent_set_result,
    verify_closure,
    verify_maximum_weight_basis,
    verify_maximum_weight_independent_set,
)

__all__ = [
    "LinearMatroid",
    "MatroidWeightFunction",
    "MatroidWeightedIntersectionResult",
    "graphic_matroid",
    "matroid_closure",
    "matroid_common_basis",
    "matroid_intersection",
    "matroid_rank",
    "maximum_weight_basis_result",
    "maximum_weight_independent_set_result",
    "verify_closure",
    "verify_common_basis_result",
    "verify_maximum_weight_basis",
    "verify_maximum_weight_independent_set",
    "verify_weighted_intersection_result",
    "weighted_intersection_certificate",
]
