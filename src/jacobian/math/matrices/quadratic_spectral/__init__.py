"""Supported exact matrix API over real quadratic fields."""

from jacobian.math.matrices.quadratic_spectral.operations import (
    inertia,
    singular_spectrum,
    symmetric_spectrum,
)
from jacobian.math.matrices.quadratic_spectral.values import (
    RealAlgebraicMultiplicity,
    RealQuadraticInertia,
    RealQuadraticSpectrum,
)

__all__ = [
    "RealAlgebraicMultiplicity",
    "RealQuadraticInertia",
    "RealQuadraticSpectrum",
    "inertia",
    "singular_spectrum",
    "symmetric_spectrum",
]
