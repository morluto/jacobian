"""Typed wire contracts for majorization and matrix mixing operations."""

from __future__ import annotations

from fractions import Fraction
from typing import Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel

MAX_DIMENSION = 20
MAX_STEPS = 500
MAX_DIGITS = 4096


def _bound_rational(value: CanonicalRational, label: str) -> None:
    require_bounded_rational(value, max_digits=MAX_DIGITS, label=label)


class RationalVector(StrictModel):
    """A finite exact rational vector with labelled coordinates."""

    labels: tuple[str, ...] = Field(min_length=1, max_length=MAX_DIMENSION)
    values: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_DIMENSION
    )

    @model_validator(mode="after")
    def require_consistent(self) -> Self:
        if len(self.labels) != len(self.values):
            raise ValueError("labels and values must have the same length")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("coordinate labels must be distinct")
        for i, v in enumerate(self.values):
            _bound_rational(v, f"values[{i}]")
        return self

    def as_fractions(self) -> tuple[Fraction, ...]:
        return tuple(v.as_fraction() for v in self.values)


class RationalMatrix(StrictModel):
    """An exact rational square matrix with row and column labels."""

    row_labels: tuple[str, ...] = Field(min_length=1, max_length=MAX_DIMENSION)
    col_labels: tuple[str, ...] = Field(min_length=1, max_length=MAX_DIMENSION)
    entries: tuple[tuple[CanonicalRational, ...], ...]

    @model_validator(mode="after")
    def require_consistent(self) -> Self:
        if len(self.row_labels) != len(self.col_labels):
            raise ValueError("row and column counts must match for a square matrix")
        if len(self.row_labels) != len(self.entries):
            raise ValueError("entry rows must match row_labels count")
        for i, row in enumerate(self.entries):
            if len(row) != len(self.col_labels):
                raise ValueError(f"row {i} has wrong column count")
            for j, v in enumerate(row):
                _bound_rational(v, f"entries[{i}][{j}]")
        if len(set(self.row_labels)) != len(self.row_labels):
            raise ValueError("row labels must be distinct")
        if len(set(self.col_labels)) != len(self.col_labels):
            raise ValueError("column labels must be distinct")
        return self

    def as_fractions(self) -> list[list[Fraction]]:
        return [[v.as_fraction() for v in row] for row in self.entries]


class MajorizationCheckRequest(StrictModel):
    """Check if x majorizes y (ordinary majorization)."""

    x: RationalVector
    y: RationalVector

    @model_validator(mode="after")
    def require_same_length(self) -> Self:
        if len(self.x.labels) != len(self.y.labels):
            raise ValueError("vectors must have the same dimension")
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
            raise ValueError("vectors must have the same dimension")
        if self.direction not in ("sub", "super"):
            raise ValueError("direction must be 'sub' or 'super'")
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
            raise ValueError("vectors must have the same dimension")
        return self


class TTransformSequenceResult(StrictModel):
    """Result of a T-transform sequence computation."""

    majorizes: bool
    steps: tuple[TTransformStep, ...]
    final_permutation: tuple[int, ...]
    intermediate_vectors: tuple[tuple[str, ...], ...]
    composed_matrix: tuple[tuple[str, ...], ...]
    target_match: bool


class DoublyStochasticCheckRequest(StrictModel):
    """Check if a rational matrix is doubly stochastic."""

    matrix: RationalMatrix


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
        fracs = self.matrix.as_fractions()
        n = len(fracs)
        for i in range(n):
            for j in range(n):
                if fracs[i][j] < 0:
                    raise ValueError(
                        "Birkhoff decomposition requires a nonnegative matrix"
                    )
        for i in range(n):
            if sum(fracs[i][j] for j in range(n)) != Fraction(1):
                raise ValueError("Birkhoff decomposition requires row sums equal to 1")
        for j in range(n):
            if sum(fracs[i][j] for i in range(n)) != Fraction(1):
                raise ValueError(
                    "Birkhoff decomposition requires column sums equal to 1"
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
            raise ValueError("eigenvalues and diagonal must have the same dimension")
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
    "RationalMatrix",
    "RationalVector",
    "SchurHornCheckRequest",
    "SchurHornCheckResult",
    "TTransformSequenceRequest",
    "TTransformSequenceResult",
    "TTransformStep",
    "WeakMajorizationCheckRequest",
    "WeakMajorizationCheckResult",
]
