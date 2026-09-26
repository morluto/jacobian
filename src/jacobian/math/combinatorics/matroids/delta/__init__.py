"""Supported native API for exact finite delta-matroids."""

from jacobian.math.combinatorics.matroids.delta.extra_ops import (
    binary,
    binary_matrix_twist,
    dual,
    minor,
)
from jacobian.math.combinatorics.matroids.delta.interlace import (
    DistanceInterlaceResult,
    distance_interlace_polynomial,
)
from jacobian.math.combinatorics.matroids.delta.operations import (
    distance_profile,
    from_feasible_sets,
    lower_matroid,
    twist,
    upper_matroid,
    verify_from_feasible_sets,
    width,
)
from jacobian.math.combinatorics.matroids.delta.relabel import relabel
from jacobian.math.combinatorics.matroids.delta.values import (
    DeltaMatroidDistanceProfile,
    FiniteDeltaMatroid,
)

__all__ = [
    "DeltaMatroidDistanceProfile",
    "DistanceInterlaceResult",
    "FiniteDeltaMatroid",
    "binary",
    "binary_matrix_twist",
    "distance_interlace_polynomial",
    "distance_profile",
    "dual",
    "from_feasible_sets",
    "lower_matroid",
    "minor",
    "relabel",
    "twist",
    "upper_matroid",
    "verify_from_feasible_sets",
    "width",
]
