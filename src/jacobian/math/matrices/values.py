"""Provider-independent exact matrix values."""

from __future__ import annotations

from fractions import Fraction
from itertools import pairwise
from typing import Any, Literal, Self

from pydantic import Field, field_validator, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalInteger,
    CanonicalRational,
)
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.canonical import parse_canonical_integer
from jacobian.math.number_theory.algebraic_numbers.quadratic import RealQuadraticValue

MAX_MATRIX_DIMENSION = 32
# The canonical dense rational matrix retains exact sources for analysis
# results whose operations admit them by their own work and result budgets,
# so its structural order is not tied to the shared computation dimension.
# The determinant operation admits square matrices through order 64.  Keep
# that complete public domain representable by the one canonical QQ matrix
# value so a determinant consumer can accept a produced matrix unchanged.
MAX_RATIONAL_MATRIX_ORDER = 64
MAX_MATRIX_SCALAR_DIGITS = MAX_CANONICAL_RATIONAL_DIGITS


def require_matrix_scalar_digits(
    entries: tuple[tuple[str | CanonicalRational, ...], ...],
    *,
    maximum: int,
    label: str,
) -> None:
    """Apply an operation-owned scalar budget to an authoritative matrix value."""

    for row in entries:
        for value in row:
            components = (value,) if isinstance(value, str) else (value.num, value.den)
            if any(len(component.lstrip("-")) > maximum for component in components):
                raise _validation_error(
                    "budget_exceeded",
                    f"{label} scalars are limited to {maximum} decimal digits",
                )


def _require_raw_matrix_envelope(
    data: object, *, maximum_axis: int, label: str
) -> object:
    """Bound raw matrix depth, axes, and scalar strings before tuple copying."""

    if not isinstance(data, dict):
        return data
    if set(data).difference({"domain", "entries"}):
        raise _validation_error("shape_mismatch", f"{label} contains unknown fields")
    entries = data.get("entries")
    if not isinstance(entries, (list, tuple)):
        return data
    if len(entries) > maximum_axis:
        raise _validation_error(
            "budget_exceeded", f"{label} has at most {maximum_axis} rows"
        )
    for row in entries:
        if not isinstance(row, (list, tuple)):
            continue
        if len(row) > maximum_axis:
            raise _validation_error(
                "budget_exceeded", f"{label} has at most {maximum_axis} columns"
            )
        for scalar in row:
            if isinstance(scalar, (list, tuple)):
                raise _validation_error(
                    "shape_mismatch", f"{label} entries must be scalar values"
                )
            if isinstance(scalar, dict) and set(scalar).difference({"num", "den"}):
                raise _validation_error(
                    "shape_mismatch", f"{label} rational scalar contains unknown fields"
                )
            components = (
                (scalar.get("num"), scalar.get("den"))
                if isinstance(scalar, dict)
                else (scalar,)
            )
            for component in components:
                if (
                    isinstance(component, (str, int))
                    and len(str(component).lstrip("-")) > MAX_MATRIX_SCALAR_DIGITS
                ):
                    raise _validation_error(
                        "budget_exceeded",
                        f"{label} scalars are limited to "
                        f"{MAX_MATRIX_SCALAR_DIGITS} decimal digits",
                    )
    return data


class RationalMatrix(StrictModel):
    """One nonempty rectangular matrix over canonical rationals."""

    domain: Literal["QQ"] = "QQ"
    entries: tuple[tuple[CanonicalRational, ...], ...] = Field(
        min_length=1,
        max_length=MAX_RATIONAL_MATRIX_ORDER,
    )

    @model_validator(mode="before")
    @classmethod
    def require_raw_matrix_envelope(cls, data: Any) -> Any:
        data = _require_raw_matrix_envelope(
            data, maximum_axis=MAX_RATIONAL_MATRIX_ORDER, label="matrix"
        )
        return canonicalize_json_containers(data)

    @model_validator(mode="after")
    def require_rectangular_nonempty_rows(self) -> Self:
        column_count = len(self.entries[0])
        if column_count == 0 or column_count > MAX_RATIONAL_MATRIX_ORDER:
            raise _validation_error(
                "budget_exceeded",
                "matrix rows must contain between 1 and "
                f"{MAX_RATIONAL_MATRIX_ORDER} entries",
            )
        if any(len(row) != column_count for row in self.entries):
            raise _validation_error(
                "budget_exceeded", "matrix rows must all have the same length"
            )
        require_matrix_scalar_digits(
            self.entries,
            maximum=MAX_MATRIX_SCALAR_DIGITS,
            label="matrix",
        )
        return self


def rational_matrix_from_fractions(
    entries: tuple[tuple[Fraction, ...], ...] | list[list[Fraction]],
) -> RationalMatrix:
    """Construct the canonical dense rational matrix from exact fractions."""

    return RationalMatrix(
        entries=tuple(
            tuple(CanonicalRational.from_fraction(value) for value in row)
            for row in entries
        )
    )


class RationalVectorSpaceBasis(StrictModel):
    """A rational vector-space basis with its ambient dimension retained.

    Unlike a dense matrix, a basis may be empty.  The explicit ambient
    dimension distinguishes the zero subspace of ``QQ^n`` for different ``n``.
    """

    domain: Literal["QQ"] = "QQ"
    ambient_dimension: int = Field(ge=1, le=MAX_RATIONAL_MATRIX_ORDER)
    vectors: tuple[tuple[CanonicalRational, ...], ...] = Field(
        default=(), max_length=MAX_RATIONAL_MATRIX_ORDER
    )

    @model_validator(mode="after")
    def require_vector_shape(self) -> Self:
        if any(len(vector) != self.ambient_dimension for vector in self.vectors):
            raise _validation_error(
                "shape_mismatch",
                "each basis vector must have the declared ambient dimension",
            )
        require_matrix_scalar_digits(
            self.vectors,
            maximum=MAX_MATRIX_SCALAR_DIGITS,
            label="basis",
        )
        return self


def rational_vector_space_basis_from_fractions(
    vectors: tuple[tuple[Fraction, ...], ...] | list[list[Fraction]],
    *,
    ambient_dimension: int,
) -> RationalVectorSpaceBasis:
    """Construct a canonical rational basis, including the empty basis."""

    return RationalVectorSpaceBasis(
        ambient_dimension=ambient_dimension,
        vectors=tuple(
            tuple(CanonicalRational.from_fraction(value) for value in vector)
            for vector in vectors
        ),
    )


class RealQuadraticMatrix(StrictModel):
    """One nonempty rectangular matrix over a shared real quadratic field."""

    domain: Literal["QQ_SQRT_D"] = "QQ_SQRT_D"
    entries: tuple[tuple[RealQuadraticValue, ...], ...] = Field(
        min_length=1,
        max_length=MAX_MATRIX_DIMENSION,
        description=(
            "Nonempty rectangular rows of a+b*sqrt(d) values. Every entry "
            "must carry the same square-free positive radicand d."
        ),
    )

    @model_validator(mode="after")
    def require_rectangular_shared_field(self) -> Self:
        column_count = len(self.entries[0])
        if column_count == 0 or column_count > MAX_MATRIX_DIMENSION:
            raise _validation_error(
                "shape_mismatch", "matrix rows must contain between 1 and 32 entries"
            )
        if any(len(row) != column_count for row in self.entries):
            raise _validation_error(
                "shape_mismatch", "matrix rows must all have the same length"
            )
        radicand = self.entries[0][0].radicand
        if any(entry.radicand != radicand for row in self.entries for entry in row):
            raise _validation_error(
                "shape_mismatch",
                "every matrix entry must belong to one shared real quadratic field",
            )
        return self


class IntegerMatrix(StrictModel):
    """One nonempty rectangular matrix over exact canonical integers."""

    domain: Literal["ZZ"] = "ZZ"
    entries: tuple[tuple[CanonicalInteger, ...], ...] = Field(
        min_length=1,
        max_length=MAX_MATRIX_DIMENSION,
    )

    @model_validator(mode="before")
    @classmethod
    def require_raw_matrix_envelope(cls, data: Any) -> Any:
        data = _require_raw_matrix_envelope(
            data, maximum_axis=MAX_MATRIX_DIMENSION, label="matrix"
        )
        return canonicalize_json_containers(data)

    @model_validator(mode="after")
    def require_rectangular_nonempty_rows(self) -> Self:
        column_count = len(self.entries[0])
        if column_count == 0 or column_count > MAX_MATRIX_DIMENSION:
            raise _validation_error(
                "budget_exceeded", "matrix rows must contain between 1 and 32 entries"
            )
        if any(len(row) != column_count for row in self.entries):
            raise _validation_error(
                "budget_exceeded", "matrix rows must all have the same length"
            )
        require_matrix_scalar_digits(
            self.entries,
            maximum=MAX_MATRIX_SCALAR_DIGITS,
            label="matrix",
        )
        return self


class SmithNormalForm(StrictModel):
    """A backend-independent positive divisibility diagonal and its metadata."""

    normal_form: IntegerMatrix
    rank: int = Field(ge=0, le=MAX_MATRIX_DIMENSION)
    invariant_factors: tuple[CanonicalInteger, ...] = Field(
        max_length=MAX_MATRIX_DIMENSION
    )
    transformation_available: Literal[False] = False
    convention: Literal["POSITIVE_DIVISIBILITY_DIAGONAL"] = (
        "POSITIVE_DIVISIBILITY_DIAGONAL"
    )

    @model_validator(mode="after")
    def require_invariant_factor_chain(self) -> Self:
        rows = len(self.normal_form.entries)
        columns = len(self.normal_form.entries[0])
        if len(self.invariant_factors) != self.rank:
            raise _validation_error(
                "shape_mismatch", "nonzero invariant factor count must equal rank"
            )
        if self.rank > min(rows, columns):
            raise _validation_error(
                "shape_mismatch", "Smith rank cannot exceed the matrix dimensions"
            )
        factors = tuple(
            parse_canonical_integer(value) for value in self.invariant_factors
        )
        if any(value <= 0 for value in factors):
            raise _validation_error(
                "shape_mismatch", "Smith invariant factors must be positive"
            )
        if any(right % left != 0 for left, right in pairwise(factors)):
            raise _validation_error(
                "shape_mismatch", "each Smith invariant factor must divide the next"
            )
        for row, entries in enumerate(self.normal_form.entries):
            for column, value in enumerate(entries):
                expected = factors[row] if row == column and row < self.rank else 0
                if parse_canonical_integer(value) != expected:
                    raise _validation_error(
                        "budget_exceeded",
                        "Smith normal form must contain its positive invariant "
                        "factors on the leading diagonal and zero elsewhere",
                    )
        return self

    @field_validator("invariant_factors")
    @classmethod
    def require_bounded_invariant_factors(
        cls, values: tuple[CanonicalInteger, ...]
    ) -> tuple[CanonicalInteger, ...]:
        for value in values:
            if len(value.lstrip("-")) > MAX_MATRIX_SCALAR_DIGITS:
                raise _validation_error(
                    "budget_exceeded",
                    f"matrix scalars are limited to {MAX_MATRIX_SCALAR_DIGITS} decimal digits",
                )
        return values


__all__ = [
    "MAX_MATRIX_DIMENSION",
    "MAX_MATRIX_SCALAR_DIGITS",
    "MAX_RATIONAL_MATRIX_ORDER",
    "IntegerMatrix",
    "RationalMatrix",
    "RationalVectorSpaceBasis",
    "RealQuadraticMatrix",
    "SmithNormalForm",
    "rational_matrix_from_fractions",
    "rational_vector_space_basis_from_fractions",
    "require_matrix_scalar_digits",
]


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"matrix.{reason}", message)
