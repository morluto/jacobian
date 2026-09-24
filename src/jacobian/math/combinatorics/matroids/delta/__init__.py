"""Supported native API for exact finite delta-matroids."""

from jacobian.math.combinatorics.matroids.delta.extra import (
    DeltaMatroidFeasibleSizeProfile,
    DeltaMatroidTwistWidthProfile,
)
from jacobian.math.combinatorics.matroids.delta.extra_ops import (
    binary,
    dual,
    feasible_size_profile,
    loop_complement,
    minor,
    twist_width_profile,
)
from jacobian.math.combinatorics.matroids.delta.operations import (
    direct_sum,
    distance,
    from_feasible_sets,
    twist,
    verify_from_feasible_sets,
    width,
)
from jacobian.math.combinatorics.matroids.delta.values import FiniteDeltaMatroid

__all__ = [
    "DeltaMatroidFeasibleSizeProfile",
    "DeltaMatroidTwistWidthProfile",
    "FiniteDeltaMatroid",
    "binary",
    "direct_sum",
    "distance",
    "dual",
    "feasible_size_profile",
    "from_feasible_sets",
    "loop_complement",
    "minor",
    "twist",
    "twist_width_profile",
    "verify_from_feasible_sets",
    "width",
]
