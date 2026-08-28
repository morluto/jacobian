"""Source-bound exact values for real-quadratic matrix spectra."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.matrices.values import RealQuadraticMatrix
from jacobian.math.number_theory.algebraic_numbers.real import RealAlgebraicValue

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
    """A complete descending 2 by 2 spectrum and its canonical source."""

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
    def require_structural_spectrum(self) -> Self:
        if len(self.matrix.entries) != 2 or len(self.matrix.entries[0]) != 2:
            raise _validation_error(
                "shape_mismatch", "spectral results retain a 2 by 2 source matrix"
            )
        if self.spectrum_kind == "SYMMETRIC_EIGENVALUES" and any(
            self.matrix.entries[row][column] != self.matrix.entries[column][row]
            for row in range(2)
            for column in range(row + 1, 2)
        ):
            raise _validation_error(
                "shape_mismatch", "symmetric spectra retain a symmetric source matrix"
            )
        if sum(row.multiplicity for row in self.values) != 2:
            raise _validation_error(
                "shape_mismatch", "2 by 2 spectral multiplicities must sum to two"
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        matrix: RealQuadraticMatrix,
        spectrum_kind: SpectrumKind,
        values: tuple[RealAlgebraicMultiplicity, ...],
    ) -> Self:
        """Construct a result emitted by the exact owner-local kernel."""

        return cls(matrix=matrix, spectrum_kind=spectrum_kind, values=values)


class RealQuadraticInertia(StrictModel):
    """Sylvester inertia and its canonical symmetric source matrix."""

    matrix: RealQuadraticMatrix
    n_positive: StrictInt = Field(ge=0, le=4)
    n_negative: StrictInt = Field(ge=0, le=4)
    n_zero: StrictInt = Field(ge=0, le=4)
    definiteness: Definiteness

    @model_validator(mode="after")
    def require_structural_inertia(self) -> Self:
        dimension = len(self.matrix.entries)
        if dimension > 4 or any(len(row) != dimension for row in self.matrix.entries):
            raise _validation_error(
                "shape_mismatch",
                "inertia results retain a square source of order at most four",
            )
        if any(
            self.matrix.entries[row][column] != self.matrix.entries[column][row]
            for row in range(dimension)
            for column in range(row + 1, dimension)
        ):
            raise _validation_error(
                "shape_mismatch", "inertia results retain a symmetric source matrix"
            )
        if self.n_positive + self.n_negative + self.n_zero != dimension:
            raise _validation_error(
                "shape_mismatch", "inertia counts must sum to the matrix dimension"
            )
        if self.definiteness != _definiteness(
            self.n_positive, self.n_negative, self.n_zero
        ):
            raise _validation_error(
                "shape_mismatch", "definiteness must agree with the inertia counts"
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        matrix: RealQuadraticMatrix,
        n_positive: int,
        n_negative: int,
        n_zero: int,
        definiteness: Definiteness,
    ) -> Self:
        """Construct a result emitted by the exact owner-local kernel."""

        return cls(
            matrix=matrix,
            n_positive=n_positive,
            n_negative=n_negative,
            n_zero=n_zero,
            definiteness=definiteness,
        )


__all__ = [
    "Definiteness",
    "RealAlgebraicMultiplicity",
    "RealQuadraticInertia",
    "RealQuadraticSpectrum",
    "SpectrumKind",
]


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"matrix.{reason}", message)


def _definiteness(positive: int, negative: int, zero: int) -> Definiteness:
    if positive == 0 and negative == 0:
        return "zero"
    if zero == 0:
        if negative == 0:
            return "positive_definite"
        if positive == 0:
            return "negative_definite"
        return "indefinite"
    if negative == 0:
        return "positive_semidefinite"
    if positive == 0:
        return "negative_semidefinite"
    return "indefinite"
