"""Exact rational coordinate-tensor operations."""

from jacobian.math.geometry.differential._models import (
    RationalLieDerivativeProfile,
)
from jacobian.math.geometry.differential.operations import (
    lie_derivative,
    verify_lie_derivative,
)
from jacobian.math.geometry.differential.values import (
    RationalCoordinateTensor,
    TensorVariance,
)

__all__ = [
    "RationalCoordinateTensor",
    "RationalLieDerivativeProfile",
    "TensorVariance",
    "lie_derivative",
    "verify_lie_derivative",
]
