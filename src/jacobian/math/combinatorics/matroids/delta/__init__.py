"""Supported native API for exact finite delta-matroids."""

from jacobian.math.combinatorics.matroids.delta.extra import (
    DeltaMatroidFeasibleSizeProfile,
    DeltaMatroidTwistWidthProfile,
)
from jacobian.math.combinatorics.matroids.delta.extra_ops import (
    binary,
    binary_matrix_twist,
    dual,
    feasible_size_profile,
    loop_complement,
    minor,
    twist_polynomial,
    twist_width_profile,
)
from jacobian.math.combinatorics.matroids.delta.interlace import (
    DistanceInterlaceResult,
    distance_interlace_polynomial,
)
from jacobian.math.combinatorics.matroids.delta.operations import (
    direct_sum,
    distance,
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
    "DeltaMatroidFeasibleSizeProfile",
    "DeltaMatroidTwistWidthProfile",
    "DistanceInterlaceResult",
    "FiniteDeltaMatroid",
    "binary",
    "binary_matrix_twist",
    "direct_sum",
    "distance",
    "distance_interlace_polynomial",
    "distance_profile",
    "dual",
    "feasible_size_profile",
    "from_feasible_sets",
    "loop_complement",
    "lower_matroid",
    "minor",
    "relabel",
    "twist",
    "twist_polynomial",
    "twist_width_profile",
    "upper_matroid",
    "verify_from_feasible_sets",
    "width",
]
