"""Typed requests for exact real-quadratic matrix spectra."""

from __future__ import annotations

from pydantic import Field

from jacobian._models import StrictModel
from jacobian.math.matrices.values import RealQuadraticMatrix


class RealQuadraticSymmetricSpectrumRequest(StrictModel):
    """One symmetric 2 by 2 matrix over a shared real quadratic field."""

    matrix: RealQuadraticMatrix = Field(
        description=(
            "An exact symmetric 2 by 2 matrix over one shared Q(sqrt(d)); "
            "the primitive spectral annihilating polynomial may use at most "
            "996 decimal digits per coefficient."
        )
    )


class RealQuadraticSingularSpectrumRequest(StrictModel):
    """One 2 by 2 matrix over a shared real quadratic field."""

    matrix: RealQuadraticMatrix = Field(
        description=(
            "An exact 2 by 2 matrix over one shared Q(sqrt(d)); the primitive "
            "singular-value annihilating polynomial may use at most 996 "
            "decimal digits per coefficient."
        )
    )


class RealQuadraticInertiaRequest(StrictModel):
    """One symmetric matrix of dimension at most sixteen over a quadratic field."""

    matrix: RealQuadraticMatrix = Field(
        description=(
            "An exact symmetric matrix of dimension at most sixteen over one "
            "shared Q(sqrt(d))."
        )
    )


__all__ = [
    "RealQuadraticInertiaRequest",
    "RealQuadraticSingularSpectrumRequest",
    "RealQuadraticSymmetricSpectrumRequest",
]
