"""Typed wire contracts for quadratic form operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_DIM = 10
MAX_ENTRY_DIGITS = 100  # limit entry magnitude to ~10^100 for bounded eigenvalue work


class SymmetricMatrix(StrictModel):
    """A symmetric integer matrix representing a quadratic form."""

    matrix: tuple[tuple[int, ...], ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_symmetric_square(self) -> Self:
        n = len(self.matrix)
        if n > MAX_DIM:
            raise ValueError(f"dimension must not exceed {MAX_DIM}")
        for row in self.matrix:
            if len(row) != n:
                raise ValueError("matrix must be square")
            for entry in row:
                if abs(entry) >= 10**MAX_ENTRY_DIGITS:
                    raise ValueError(
                        f"matrix entries must not exceed {MAX_ENTRY_DIGITS} digits"
                    )
        for i in range(n):
            for j in range(n):
                if self.matrix[i][j] != self.matrix[j][i]:
                    raise ValueError("matrix must be symmetric")
        return self


class EvaluationRequest(StrictModel):
    """Evaluate q(x) = x^T A x for an integer vector x."""

    form: SymmetricMatrix
    vector: tuple[int, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_matching_dimension(self) -> Self:
        if len(self.vector) != len(self.form.matrix):
            raise ValueError("vector dimension must match form dimension")
        return self


class DiscriminantRequest(StrictModel):
    """Compute the discriminant of a quadratic form."""

    form: SymmetricMatrix


class SignatureRequest(StrictModel):
    """Compute the signature (n_pos, n_neg, n_zero) of a quadratic form."""

    form: SymmetricMatrix


class EvaluationResult(StrictModel):
    """The value q(x) = x^T A x."""

    value: int
    dimension: int


class DiscriminantResult(StrictModel):
    """The discriminant det(A) of a quadratic form."""

    discriminant: int
    dimension: int


class SignatureResult(StrictModel):
    """The inertia (positive, negative, zero eigenvalue counts) of a form."""

    n_positive: int
    n_negative: int
    n_zero: int
    is_positive_definite: bool
    is_negative_definite: bool
    is_indefinite: bool
