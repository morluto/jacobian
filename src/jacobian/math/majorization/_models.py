"""Typed wire contracts for majorization and matrix mixing operations."""

from __future__ import annotations

from fractions import Fraction
from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel
from jacobian.math.matrices.values import RationalMatrix

MAX_DIMENSION = 20
MAX_STEPS = 500
MAX_DIGITS = 4096


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"majorization.{reason}", message)


def _bound_rational(value: CanonicalRational, label: str) -> None:
    try:
        require_bounded_rational(value, max_digits=MAX_DIGITS, label=label)
    except ValueError as error:
        raise _validation_error("rational_bound", str(error)) from error


class RationalVector(StrictModel):
    """A finite exact rational vector with labelled coordinates."""

    labels: tuple[str, ...] = Field(min_length=1, max_length=MAX_DIMENSION)
    values: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_DIMENSION
    )

    @model_validator(mode="after")
    def require_consistent(self) -> Self:
        if len(self.labels) != len(self.values):
            raise _validation_error(
                "vector_length", "labels and values must have the same length"
            )
        if len(set(self.labels)) != len(self.labels):
            raise _validation_error(
                "duplicate_label", "coordinate labels must be distinct"
            )
        for i, v in enumerate(self.values):
            _bound_rational(v, f"values[{i}]")
        return self

    def as_fractions(self) -> tuple[Fraction, ...]:
        return tuple(v.as_fraction() for v in self.values)


def _require_majorization_matrix(matrix: RationalMatrix) -> None:
    """Apply majorization's square-matrix and scalar admission envelope."""

    if len(matrix.entries) > MAX_DIMENSION:
        raise _validation_error(
            "matrix_dimension", f"matrix order must not exceed {MAX_DIMENSION}"
        )
    if len(matrix.entries) != len(matrix.entries[0]):
        raise _validation_error(
            "matrix_not_square", "majorization requires a square matrix"
        )
    for row_index, row in enumerate(matrix.entries):
        for column_index, value in enumerate(row):
            _bound_rational(value, f"entries[{row_index}][{column_index}]")


class MajorizationCheckRequest(StrictModel):
    """Check if x majorizes y (ordinary majorization)."""

    x: RationalVector
    y: RationalVector

    @model_validator(mode="after")
    def require_same_length(self) -> Self:
        if len(self.x.labels) != len(self.y.labels):
            raise _validation_error(
                "vector_dimension", "vectors must have the same dimension"
            )
        return self


class MajorizationCheckResult(StrictModel):
    """Result of a majorization check."""

    majorizes: bool
    total_sum_match: bool
    prefix_slacks: tuple[str, ...]
    first_failed_prefix: int | None = None


class WeakMajorizationCheckRequest(StrictModel):
    """Check weak majorization."""

    x: RationalVector
    y: RationalVector
    direction: str = Field(default="sub")

    @model_validator(mode="after")
    def require_same_length(self) -> Self:
        if len(self.x.labels) != len(self.y.labels):
            raise _validation_error(
                "vector_dimension", "vectors must have the same dimension"
            )
        if self.direction not in ("sub", "super"):
            raise _validation_error("direction", "direction must be 'sub' or 'super'")
        return self


class WeakMajorizationCheckResult(StrictModel):
    """Result of a weak majorization check."""

    holds: bool
    direction: str
    prefix_slack: tuple[str, ...]
    first_failed_prefix: int | None = None


class TTransformStep(StrictModel):
    """One T-transform step."""

    i_label: str
    j_label: str
    lam: CanonicalRational


class TTransformSequenceRequest(StrictModel):
    """Compute a T-transform sequence from x to y."""

    x: RationalVector
    y: RationalVector

    @model_validator(mode="after")
    def require_same_length(self) -> Self:
        if len(self.x.labels) != len(self.y.labels):
            raise _validation_error(
                "vector_dimension", "vectors must have the same dimension"
            )
        return self


class TTransformSequenceResult(StrictModel):
    """Result of a T-transform sequence computation."""

    majorizes: bool
    steps: tuple[TTransformStep, ...]
    final_permutation: tuple[int, ...]
    intermediate_vectors: tuple[tuple[str, ...], ...]
    composed_matrix: tuple[tuple[str, ...], ...]
    target_match: bool

    @model_validator(mode="after")
    def bind_majorization_outcome(self) -> Self:
        if self.majorizes and not self.target_match:
            raise _validation_error(
                "t_transform_target_mismatch",
                "a positive T-transform construction must reach the target",
            )
        if not self.majorizes and (
            self.steps
            or self.final_permutation
            or self.intermediate_vectors
            or self.composed_matrix
            or self.target_match
        ):
            raise _validation_error(
                "negative_t_transform_shape",
                "a negative majorization result has no construction witness",
            )
        return self


class DoublyStochasticCheckRequest(StrictModel):
    """Check if a rational matrix is doubly stochastic."""

    matrix: RationalMatrix

    @model_validator(mode="after")
    def require_admitted_matrix(self) -> Self:
        _require_majorization_matrix(self.matrix)
        return self


class DoublyStochasticCheckResult(StrictModel):
    """Result of a doubly stochastic check."""

    is_doubly_stochastic: bool
    row_sums: tuple[str, ...]
    col_sums: tuple[str, ...]
    first_negative_entry: tuple[int, int] | None = None
    first_bad_row: int | None = None
    first_bad_col: int | None = None


class BirkhoffTerm(StrictModel):
    """One term in a Birkhoff decomposition."""

    weight: CanonicalRational
    permutation: tuple[int, ...]


class BirkhoffDecompositionRequest(StrictModel):
    """Compute a Birkhoff-von Neumann decomposition of a doubly stochastic matrix."""

    matrix: RationalMatrix

    @model_validator(mode="after")
    def require_doubly_stochastic(self) -> Self:
        _require_majorization_matrix(self.matrix)
        fracs = tuple(
            tuple(value.as_fraction() for value in row) for row in self.matrix.entries
        )
        n = len(fracs)
        for i in range(n):
            for j in range(n):
                if fracs[i][j] < 0:
                    raise _validation_error(
                        "birkhoff_negative_entry",
                        "Birkhoff decomposition requires a nonnegative matrix",
                    )
        for i in range(n):
            if sum(fracs[i][j] for j in range(n)) != Fraction(1):
                raise _validation_error(
                    "birkhoff_row_sum",
                    "Birkhoff decomposition requires row sums equal to 1",
                )
        for j in range(n):
            if sum(fracs[i][j] for i in range(n)) != Fraction(1):
                raise _validation_error(
                    "birkhoff_column_sum",
                    "Birkhoff decomposition requires column sums equal to 1",
                )
        return self


class BirkhoffDecompositionResult(StrictModel):
    """Result of a Birkhoff decomposition."""

    terms: tuple[BirkhoffTerm, ...]
    weights_sum: str
    reconstruction_matches: bool


class SchurHornCheckRequest(StrictModel):
    """Check Schur-Horn feasibility."""

    eigenvalues: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_DIMENSION
    )
    diagonal: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_DIMENSION
    )

    @model_validator(mode="after")
    def require_consistent(self) -> Self:
        if len(self.eigenvalues) != len(self.diagonal):
            raise _validation_error(
                "schur_horn_dimension",
                "eigenvalues and diagonal must have the same dimension",
            )
        for i, v in enumerate(self.eigenvalues):
            _bound_rational(v, f"eigenvalues[{i}]")
        for i, v in enumerate(self.diagonal):
            _bound_rational(v, f"diagonal[{i}]")
        return self


class SchurHornCheckResult(StrictModel):
    """Result of Schur-Horn feasibility check."""

    feasible: bool
    eigenvalues_sorted: tuple[str, ...]
    diagonal_sorted: tuple[str, ...]
    prefix_slack: tuple[str, ...]
    first_failed_prefix: int | None = None
    total_sum_match: bool


__all__ = [
    "BirkhoffDecompositionRequest",
    "BirkhoffDecompositionResult",
    "BirkhoffTerm",
    "DoublyStochasticCheckRequest",
    "DoublyStochasticCheckResult",
    "MajorizationCheckRequest",
    "MajorizationCheckResult",
    "RationalVector",
    "SchurHornCheckRequest",
    "SchurHornCheckResult",
    "TTransformSequenceRequest",
    "TTransformSequenceResult",
    "TTransformStep",
    "WeakMajorizationCheckRequest",
    "WeakMajorizationCheckResult",
]
