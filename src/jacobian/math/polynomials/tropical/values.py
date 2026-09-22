"""Canonical exact values for bounded tropical algebra."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel
from jacobian.math._labels import OpaqueLabel

MAX_TROPICAL_SCALAR_DIGITS = 8_192
MAX_TROPICAL_VECTOR_DIMENSION = 128
MAX_TROPICAL_MATRIX_CELLS = 4_096
MAX_TROPICAL_POLYNOMIAL_TERMS = 512
MAX_TROPICAL_EXPONENT = 1_024


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"tropical.{reason}", message)


class TropicalSemiring(StrictModel):
    convention: Literal["MIN_PLUS", "MAX_PLUS"]
    base: Literal["ZZ", "QQ"]


class TropicalScalar(StrictModel):
    semiring: TropicalSemiring
    kind: Literal["FINITE", "POSITIVE_INFINITY", "NEGATIVE_INFINITY"]
    value: CanonicalRational | None = None

    @model_validator(mode="after")
    def require_licensed_scalar(self) -> Self:
        if self.kind == "FINITE":
            if self.value is None:
                raise _validation_error(
                    "finite_missing_value", "a finite scalar must carry its value"
                )
            if self.semiring.base == "ZZ" and self.value.den != 1:
                raise _validation_error(
                    "nonintegral_integer_scalar",
                    "a ZZ tropical scalar must be integral",
                )
        else:
            if self.value is not None:
                raise _validation_error(
                    "infinite_carries_value", "an infinite scalar carries no value"
                )
            if (self.kind == "POSITIVE_INFINITY") != (
                self.semiring.convention == "MIN_PLUS"
            ):
                raise _validation_error(
                    "unlicensed_infinity",
                    "only the semiring-licensed infinity is an element",
                )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        semiring: TropicalSemiring,
        kind: Literal["FINITE", "POSITIVE_INFINITY", "NEGATIVE_INFINITY"],
        value: CanonicalRational | None,
    ) -> Self:
        return cls.model_construct(semiring=semiring, kind=kind, value=value)


def require_scalar_budget(scalar: TropicalScalar) -> None:
    if scalar.value is not None:
        try:
            require_bounded_rational(
                scalar.value,
                max_digits=MAX_TROPICAL_SCALAR_DIGITS,
                label="tropical scalar",
            )
        except ValueError as error:
            raise _validation_error("scalar_budget", str(error)) from error


class TropicalVector(StrictModel):
    semiring: TropicalSemiring
    axis: tuple[OpaqueLabel, ...] = Field(max_length=MAX_TROPICAL_VECTOR_DIMENSION)
    entries: tuple[TropicalScalar, ...] = Field(
        max_length=MAX_TROPICAL_VECTOR_DIMENSION
    )

    @model_validator(mode="after")
    def require_shape(self) -> Self:
        if len(set(self.axis)) != len(self.axis) or len(self.axis) != len(self.entries):
            raise _validation_error(
                "vector_shape",
                "vector axis and entries must have the same unique labels",
            )
        if any(entry.semiring != self.semiring for entry in self.entries):
            raise _validation_error(
                "vector_semiring", "every vector entry must carry the vector semiring"
            )
        return self


class TropicalMatrix(StrictModel):
    semiring: TropicalSemiring
    row_axis: tuple[OpaqueLabel, ...] = Field(max_length=MAX_TROPICAL_VECTOR_DIMENSION)
    column_axis: tuple[OpaqueLabel, ...] = Field(
        max_length=MAX_TROPICAL_VECTOR_DIMENSION
    )
    entries: tuple[tuple[TropicalScalar, ...], ...]

    @model_validator(mode="after")
    def require_shape(self) -> Self:
        if len(set(self.row_axis)) != len(self.row_axis) or len(
            set(self.column_axis)
        ) != len(self.column_axis):
            raise _validation_error(
                "matrix_axis", "matrix axes must have unique labels"
            )
        if len(self.entries) != len(self.row_axis) or any(
            len(row) != len(self.column_axis) for row in self.entries
        ):
            raise _validation_error(
                "matrix_shape", "matrix entries must match row and column axes"
            )
        if any(
            entry.semiring != self.semiring for row in self.entries for entry in row
        ):
            raise _validation_error(
                "matrix_semiring", "every matrix entry must carry the matrix semiring"
            )
        if len(self.row_axis) * len(self.column_axis) > MAX_TROPICAL_MATRIX_CELLS:
            raise _validation_error(
                "matrix_budget", "tropical matrix has too many cells"
            )
        return self


class TropicalPolynomialTerm(StrictModel):
    exponents: tuple[int, ...]
    coefficient: TropicalScalar

    @model_validator(mode="after")
    def require_exponents(self) -> Self:
        if any(value < 0 or value > MAX_TROPICAL_EXPONENT for value in self.exponents):
            raise _validation_error(
                "exponent_bound", "exponents must be nonnegative and bounded"
            )
        return self


class TropicalPolynomial(StrictModel):
    semiring: TropicalSemiring
    variables: tuple[OpaqueLabel, ...] = Field(max_length=MAX_TROPICAL_VECTOR_DIMENSION)
    terms: tuple[TropicalPolynomialTerm, ...] = Field(
        max_length=MAX_TROPICAL_POLYNOMIAL_TERMS
    )

    @model_validator(mode="after")
    def require_canonical_terms(self) -> Self:
        if len(set(self.variables)) != len(self.variables):
            raise _validation_error(
                "polynomial_axis", "polynomial variables must be unique"
            )
        exponents = tuple(term.exponents for term in self.terms)
        if any(len(exp) != len(self.variables) for exp in exponents):
            raise _validation_error(
                "polynomial_shape", "term exponents must match variables"
            )
        if any(term.coefficient.semiring != self.semiring for term in self.terms):
            raise _validation_error(
                "polynomial_semiring", "terms must carry the polynomial semiring"
            )
        if exponents != tuple(sorted(exponents)) or len(set(exponents)) != len(
            exponents
        ):
            raise _validation_error(
                "polynomial_terms",
                "polynomial terms must be sorted with unique exponents",
            )
        if any(term.coefficient.kind != "FINITE" for term in self.terms):
            raise _validation_error(
                "polynomial_infinity",
                "infinite coefficients are omitted from canonical polynomials",
            )
        return self


__all__ = [
    "MAX_TROPICAL_EXPONENT",
    "MAX_TROPICAL_MATRIX_CELLS",
    "MAX_TROPICAL_POLYNOMIAL_TERMS",
    "MAX_TROPICAL_SCALAR_DIGITS",
    "MAX_TROPICAL_VECTOR_DIMENSION",
    "TropicalMatrix",
    "TropicalPolynomial",
    "TropicalPolynomialTerm",
    "TropicalScalar",
    "TropicalSemiring",
    "TropicalVector",
    "require_scalar_budget",
]
