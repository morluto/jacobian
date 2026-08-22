"""Typed wire contracts for cluster algebra operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_EXCHANGE_SIZE = 16
# Admitted seeds carry bounded integer coefficients: entries and symmetrizer
# entries stay below 10**MAX_EXCHANGE_ENTRY_DIGITS. Mutation forms
# b_ij + b_ik * b_kj, whose magnitude reaches 10**(2 * D) - 10**D for admitted
# entries below 10**D, so results admit that derived bound plus one digit of
# slack while every skew-symmetrizability product stays bounded work.
MAX_EXCHANGE_ENTRY_DIGITS = 64
MAX_MUTATED_ENTRY_DIGITS = 2 * MAX_EXCHANGE_ENTRY_DIGITS + 1
_MAX_INPUT_ENTRY_MAGNITUDE = 10**MAX_EXCHANGE_ENTRY_DIGITS


def _require_bounded_entries(matrix: ExchangeMatrix, *, max_digits: int) -> None:
    magnitude = 10**max_digits
    if any(
        abs(entry) >= magnitude
        for row in matrix.entries
        for entry in row
    ):
        raise ValueError(
            f"exchange-matrix coefficients exceed the {max_digits}-digit bound"
        )


def _require_bounded_symmetrizer(matrix: ExchangeMatrix) -> None:
    if any(abs(d) >= _MAX_INPUT_ENTRY_MAGNITUDE for d in matrix.symmetrizer):
        raise ValueError(
            "symmetrizer coefficients exceed the "
            f"{MAX_EXCHANGE_ENTRY_DIGITS}-digit bound"
        )


def _require_input_seed(matrix: ExchangeMatrix) -> None:
    """Narrow admitted seeds so mutation output stays within the result bound."""
    _require_bounded_entries(matrix, max_digits=MAX_EXCHANGE_ENTRY_DIGITS)


def _require_shape(matrix: ExchangeMatrix) -> None:
    if len(matrix.entries) != matrix.n:
        raise ValueError("entries must be an n x n matrix")
    for row in matrix.entries:
        if len(row) != matrix.n:
            raise ValueError("entries must be a square matrix")
    if len(matrix.symmetrizer) != matrix.n:
        raise ValueError("symmetrizer must have n entries")


class ExchangeMatrix(StrictModel):
    """A skew-symmetrizable integer exchange matrix B.

    The symmetrizer D must have strictly positive diagonal entries: a
    diagonal matrix with positive diagonal satisfying DB = -B^T is exactly
    what makes B an exchange matrix, and a zero or negative entry would
    accept matrices that are not skew-symmetrizable.
    """

    n: int = Field(ge=1, le=MAX_EXCHANGE_SIZE)
    entries: tuple[tuple[int, ...], ...] = Field(min_length=1, max_length=MAX_EXCHANGE_SIZE)
    symmetrizer: tuple[int, ...] = Field(min_length=1, max_length=MAX_EXCHANGE_SIZE)

    @model_validator(mode="after")
    def require_valid_matrix(self) -> Self:
        # The representation ceiling keeps every skew-symmetrizability product
        # bounded; admitted requests narrow this further via _require_input_seed.
        _require_bounded_entries(
            self, max_digits=MAX_MUTATED_ENTRY_DIGITS
        )
        _require_bounded_symmetrizer(self)
        _require_shape(self)
        for i in range(self.n):
            if self.symmetrizer[i] <= 0:
                raise ValueError(
                    "symmetrizer entries must be strictly positive integers"
                )
        for i in range(self.n):
            if self.entries[i][i] != 0:
                raise ValueError("diagonal entries must be zero")
        for i in range(self.n):
            for j in range(self.n):
                if self.symmetrizer[i] * self.entries[i][j] != -self.symmetrizer[j] * self.entries[j][i]:
                    raise ValueError(
                        f"skew-symmetrizability condition violated at ({i}, {j})"
                    )
        return self


class SeedMutationRequest(StrictModel):
    """Mutate a cluster seed at a specified mutable index."""

    exchange_matrix: ExchangeMatrix
    mutation_index: int = Field(ge=0)

    @model_validator(mode="after")
    def require_valid_index(self) -> Self:
        if self.mutation_index >= self.exchange_matrix.n:
            raise ValueError("mutation_index must be in 0..n-1")
        _require_input_seed(self.exchange_matrix)
        return self


class SeedMutationResult(StrictModel):
    """The mutated exchange matrix after applying the Fomin-Zelevinsky mutation."""

    exchange_matrix: ExchangeMatrix
    mutation_index: int = Field(ge=0)


class GVectorRequest(StrictModel):
    """Compute the g-vector matrix for principal coefficients."""

    exchange_matrix: ExchangeMatrix

    @model_validator(mode="after")
    def require_admissible_seed(self) -> Self:
        _require_input_seed(self.exchange_matrix)
        return self


def _identity_matrix(n: int) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(1 if i == j else 0 for j in range(n)) for i in range(n))


class GVectorResult(StrictModel):
    """The g-vector matrix (identity for the initial seed).

    Retains the source seed so an exact result must carry its identity matrix
    under the single fixed convention.
    """

    exchange_matrix: ExchangeMatrix
    g_matrix: tuple[tuple[int, ...], ...]
    convention: Literal["FOMIN_ZELEVINSKY"] = "FOMIN_ZELEVINSKY"

    @model_validator(mode="after")
    def require_initial_g_vectors(self) -> Self:
        if self.g_matrix != _identity_matrix(self.exchange_matrix.n):
            raise ValueError(
                "g_matrix must be the n x n identity of the source exchange matrix"
            )
        return self


__all__ = [
    "ExchangeMatrix",
    "GVectorRequest",
    "GVectorResult",
    "SeedMutationRequest",
    "SeedMutationResult",
]
