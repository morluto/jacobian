"""Typed requests for exact real-quadratic matrix spectra."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

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

    @model_validator(mode="after")
    def require_admitted_matrix(self) -> Self:
        from jacobian.math.matrices.quadratic_spectral.operations import (
            require_symmetric_spectrum_matrix,
        )

        require_symmetric_spectrum_matrix(self.matrix)
        return self


class RealQuadraticSingularSpectrumRequest(StrictModel):
    """One 2 by 2 matrix over a shared real quadratic field."""

    matrix: RealQuadraticMatrix = Field(
        description=(
            "An exact 2 by 2 matrix over one shared Q(sqrt(d)); the primitive "
            "singular-value annihilating polynomial may use at most 996 "
            "decimal digits per coefficient."
        )
    )

    @model_validator(mode="after")
    def require_admitted_matrix(self) -> Self:
        from jacobian.math.matrices.quadratic_spectral.operations import (
            require_singular_spectrum_matrix,
        )

        require_singular_spectrum_matrix(self.matrix)
        return self


class RealQuadraticInertiaRequest(StrictModel):
    """One symmetric matrix of dimension at most four over a quadratic field."""

    matrix: RealQuadraticMatrix = Field(
        description=(
            "An exact symmetric matrix of dimension at most four over one "
            "shared Q(sqrt(d))."
        )
    )

    @model_validator(mode="after")
    def require_admitted_matrix(self) -> Self:
        from jacobian.math.matrices.quadratic_spectral.operations import (
            require_inertia_matrix,
        )

        require_inertia_matrix(self.matrix)
        return self


__all__ = [
    "RealQuadraticInertiaRequest",
    "RealQuadraticSingularSpectrumRequest",
    "RealQuadraticSymmetricSpectrumRequest",
]
