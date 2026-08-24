"""Source-bound exact values for real-quadratic matrix spectra."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._models import StrictModel
from jacobian.math.matrices.values import RealQuadraticMatrix
from jacobian.math.real_algebraic import RealAlgebraicValue

SpectrumKind = Literal["SYMMETRIC_EIGENVALUES", "SINGULAR_VALUES"]
Definiteness = Literal[
    "positive_definite",
    "positive_semidefinite",
    "negative_definite",
    "negative_semidefinite",
    "zero",
    "indefinite",
]


class RealAlgebraicMultiplicity(StrictModel):
    """One exact real algebraic value and its spectral multiplicity."""

    value: RealAlgebraicValue
    multiplicity: StrictInt = Field(ge=1, le=2)


class RealQuadraticSpectrum(StrictModel):
    """A complete descending 2 by 2 spectrum bound to its source matrix."""

    matrix: RealQuadraticMatrix
    spectrum_kind: SpectrumKind
    values: tuple[RealAlgebraicMultiplicity, ...] = Field(
        min_length=1,
        max_length=2,
        description=(
            "Distinct exact values in descending real order, with multiplicities "
            "summing to two."
        ),
    )
    ordering: Literal["DESCENDING"] = "DESCENDING"

    @model_validator(mode="after")
    def bind_complete_spectrum(self) -> Self:
        from jacobian.math.matrices.quadratic_spectral.operations import spectrum_rows

        if sum(row.multiplicity for row in self.values) != 2:
            raise ValueError("2 by 2 spectral multiplicities must sum to two")
        expected = spectrum_rows(self.matrix, self.spectrum_kind)
        if self.values != expected:
            raise ValueError("spectrum does not match the exact source matrix")
        return self


class RealQuadraticInertia(StrictModel):
    """Sylvester inertia bound to a symmetric real-quadratic matrix."""

    matrix: RealQuadraticMatrix
    n_positive: StrictInt = Field(ge=0, le=4)
    n_negative: StrictInt = Field(ge=0, le=4)
    n_zero: StrictInt = Field(ge=0, le=4)
    definiteness: Definiteness

    @model_validator(mode="after")
    def bind_exact_inertia(self) -> Self:
        from jacobian.math.matrices.quadratic_spectral.operations import inertia_data

        expected = inertia_data(self.matrix)
        actual = (
            self.n_positive,
            self.n_negative,
            self.n_zero,
            self.definiteness,
        )
        if actual != expected:
            raise ValueError("inertia does not match the exact source matrix")
        return self


__all__ = [
    "Definiteness",
    "RealAlgebraicMultiplicity",
    "RealQuadraticInertia",
    "RealQuadraticSpectrum",
    "SpectrumKind",
]
