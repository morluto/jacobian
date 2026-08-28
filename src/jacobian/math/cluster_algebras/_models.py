"""Typed wire contracts for cluster algebra operations."""

from __future__ import annotations

from math import isqrt
from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer
from jacobian.canonical import parse_canonical_integer as _parse_int

# Exchange-matrix values carry bounded integers so every skew-symmetrizability
# product stays bounded work: symmetrizer entries stay below
# 10**MAX_EXCHANGE_ENTRY_DIGITS and entries below 10**MAX_MUTATED_ENTRY_DIGITS.
# Mutation forms b_ij + [b_ik]_+[b_kj]_+ - [-b_ik]_+[-b_kj]_+, whose magnitude
# reaches roughly 10**(2 * D) for entries below 10**D, so no static seed cap
# can admit its own results: SeedMutationRequest instead derives the exact
# one-step growth of its matrix at the requested index and admits only seeds
# whose mutated result stays within the representation ceiling. Every returned
# mutation result is therefore itself an admissible input.
MAX_EXCHANGE_ENTRY_DIGITS = 64
MAX_MUTATED_ENTRY_DIGITS = 2 * MAX_EXCHANGE_ENTRY_DIGITS + 1
# Work and output derive from dimensions and coefficient heights rather than a
# coarse rank cap: mutation performs n**2 updates on coefficients of at most
# MAX_MUTATED_ENTRY_DIGITS digits, validation performs n**2 symmetrized
# products, and the exact result is an n x n matrix with cells of at most
# MAX_MUTATED_ENTRY_DIGITS digits. The cell budget below therefore bounds all
# admitted kernel work and serialized output; it is a conservative envelope,
# not a mathematical restriction on exchange-matrix dimension.
MAX_EXCHANGE_CELLS = 4096
_MAX_EXCHANGE_SIDE = isqrt(MAX_EXCHANGE_CELLS)
_MAX_ENTRY_STRING_LENGTH = MAX_MUTATED_ENTRY_DIGITS + 1
_MAX_SYMMETRIZER_STRING_LENGTH = MAX_EXCHANGE_ENTRY_DIGITS + 1
_MAX_ENTRY_MAGNITUDE = 10**MAX_MUTATED_ENTRY_DIGITS


def _validation_error(code: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(code, message)


ExchangeCoefficient = Annotated[
    CanonicalInteger,
    StringConstraints(strict=True, max_length=_MAX_ENTRY_STRING_LENGTH),
]
SymmetrizerCoefficient = Annotated[
    CanonicalInteger,
    StringConstraints(strict=True, max_length=_MAX_SYMMETRIZER_STRING_LENGTH),
]


def parsed_entries(matrix: ExchangeMatrix) -> tuple[tuple[int, ...], ...]:
    """The exchange matrix as exact integers for kernel consumption."""
    return tuple(tuple(_parse_int(value) for value in row) for row in matrix.entries)


def parsed_symmetrizer(matrix: ExchangeMatrix) -> tuple[int, ...]:
    return tuple(_parse_int(d) for d in matrix.symmetrizer)


def encoded_entries(
    entries: tuple[tuple[int, ...], ...],
) -> tuple[tuple[str, ...], ...]:
    """Canonical integer strings so every coefficient survives JSON transport."""
    return tuple(
        tuple(format_canonical_integer(value) for value in row) for row in entries
    )


def _require_bounded_entries(matrix: ExchangeMatrix, *, max_digits: int) -> None:
    # Inspect canonical-string lengths before any integer conversion so the
    # declared digit ceiling bounds parsing work and intermediate allocation.
    if any(
        len(value.lstrip("-")) > max_digits for row in matrix.entries for value in row
    ):
        raise _validation_error(
            "cluster_algebra.exchange_entries_bounded",
            f"exchange-matrix coefficients exceed the {max_digits}-digit bound",
        )


def _require_bounded_symmetrizer(matrix: ExchangeMatrix) -> None:
    if any(len(d.lstrip("-")) > MAX_EXCHANGE_ENTRY_DIGITS for d in matrix.symmetrizer):
        raise _validation_error(
            "cluster_algebra.symmetrizer_bounded",
            "symmetrizer coefficients exceed the "
            f"{MAX_EXCHANGE_ENTRY_DIGITS}-digit bound",
        )


def _require_mutatable(matrix: ExchangeMatrix, index: int) -> None:
    """Admit exactly those mutations whose result stays representable."""
    rows = parsed_entries(matrix)
    pivot_row = rows[index]
    for i, row in enumerate(rows):
        positive_i = max(row[index], 0)
        negative_i = max(-row[index], 0)
        for j, b_ij in enumerate(row):
            if i == index or j == index:
                continue
            b_kj = pivot_row[j]
            delta = positive_i * max(b_kj, 0) - negative_i * max(-b_kj, 0)
            if abs(b_ij + delta) >= _MAX_ENTRY_MAGNITUDE:
                raise _validation_error(
                    "cluster_algebra.mutation_bounded",
                    "mutation result exceeds the "
                    f"{MAX_MUTATED_ENTRY_DIGITS}-digit exchange-matrix bound",
                )


def _require_shape(matrix: ExchangeMatrix) -> None:
    if len(matrix.entries) != matrix.n:
        raise _validation_error(
            "cluster_algebra.entries_shape", "entries must be an n x n matrix"
        )
    for row in matrix.entries:
        if len(row) != matrix.n:
            raise _validation_error(
                "cluster_algebra.entries_square", "entries must be a square matrix"
            )
    if len(matrix.symmetrizer) != matrix.n:
        raise _validation_error(
            "cluster_algebra.symmetrizer_shape", "symmetrizer must have n entries"
        )


class ExchangeMatrix(StrictModel):
    """A skew-symmetrizable integer exchange matrix B.

    The symmetrizer D must have strictly positive diagonal entries: a
    diagonal matrix with positive diagonal satisfying DB = -B^T is exactly
    what makes B an exchange matrix, and a zero or negative entry would
    accept matrices that are not skew-symmetrizable.
    """

    # The side limit is isqrt(MAX_EXCHANGE_CELLS): admitting n admits exactly
    # n**2 <= MAX_EXCHANGE_CELLS cells of bounded-coefficient work.
    n: int = Field(ge=1, le=_MAX_EXCHANGE_SIDE)
    # Coefficients are canonical integer strings: mutation squares magnitudes,
    # so raw JSON integers would leave the interoperable transport range while
    # the mathematical value stays exact. The per-string max_length is the
    # schema-visible form of the digit ceiling (digits plus an optional sign).
    entries: tuple[tuple[ExchangeCoefficient, ...], ...] = Field(
        min_length=1, max_length=_MAX_EXCHANGE_SIDE
    )
    symmetrizer: tuple[SymmetrizerCoefficient, ...] = Field(
        min_length=1, max_length=_MAX_EXCHANGE_SIDE
    )

    @model_validator(mode="after")
    def require_valid_matrix(self) -> Self:
        # The representation ceiling keeps every skew-symmetrizability product
        # bounded; mutation requests further derive their one-step growth via
        # _require_mutatable.
        _require_shape(self)
        _require_bounded_entries(self, max_digits=MAX_MUTATED_ENTRY_DIGITS)
        _require_bounded_symmetrizer(self)
        entries = parsed_entries(self)
        symmetrizer = parsed_symmetrizer(self)
        for i in range(self.n):
            if symmetrizer[i] <= 0:
                raise _validation_error(
                    "cluster_algebra.symmetrizer_positive",
                    "symmetrizer entries must be strictly positive integers",
                )
        for i in range(self.n):
            if entries[i][i] != 0:
                raise _validation_error(
                    "cluster_algebra.diagonal_zero", "diagonal entries must be zero"
                )
        for i in range(self.n):
            for j in range(self.n):
                if symmetrizer[i] * entries[i][j] != -symmetrizer[j] * entries[j][i]:
                    raise _validation_error(
                        "cluster_algebra.skew_symmetrizable",
                        f"skew-symmetrizability condition violated at ({i}, {j})",
                    )
        return self


class SeedMutationRequest(StrictModel):
    """Mutate a cluster seed at a specified mutable index."""

    exchange_matrix: ExchangeMatrix
    mutation_index: int = Field(ge=0)


class SeedMutationResult(StrictModel):
    """The mutated exchange matrix after applying the Fomin-Zelevinsky mutation."""

    source_exchange_matrix: ExchangeMatrix
    exchange_matrix: ExchangeMatrix
    mutation_index: int = Field(ge=0)

    @model_validator(mode="after")
    def require_result_shape(self) -> Self:
        if self.mutation_index >= self.exchange_matrix.n:
            raise _validation_error(
                "cluster_algebra.mutation_index", "mutation_index must be in 0..n-1"
            )
        if self.mutation_index >= self.source_exchange_matrix.n:
            raise _validation_error(
                "cluster_algebra.mutation_index", "mutation_index must be in 0..n-1"
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        source_exchange_matrix: ExchangeMatrix,
        mutation_index: int,
        *,
        exchange_matrix: ExchangeMatrix,
    ) -> Self:
        """Construct a result emitted by the owner-local mutation kernel."""

        return cls.model_construct(
            source_exchange_matrix=source_exchange_matrix,
            exchange_matrix=exchange_matrix,
            mutation_index=mutation_index,
        )


class GVectorRequest(StrictModel):
    """Compute the g-vector matrix for principal coefficients.

    The result is the identity matrix of the seed, so the exchange matrix's
    own representation ceilings bound all accepted work.
    """

    exchange_matrix: ExchangeMatrix


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
    def require_g_matrix_shape(self) -> Self:
        n = self.exchange_matrix.n
        if len(self.g_matrix) != n or any(len(row) != n for row in self.g_matrix):
            raise _validation_error(
                "cluster_algebra.g_matrix_shape",
                "g_matrix must be an n x n matrix for the source exchange matrix",
            )
        return self

    @classmethod
    def _from_kernel(cls, exchange_matrix: ExchangeMatrix) -> Self:
        """Construct initial-seed g-vectors emitted by the owner-local kernel."""

        return cls.model_construct(
            exchange_matrix=exchange_matrix,
            g_matrix=_identity_matrix(exchange_matrix.n),
            convention="FOMIN_ZELEVINSKY",
        )


__all__ = [
    "ExchangeMatrix",
    "GVectorRequest",
    "GVectorResult",
    "SeedMutationRequest",
    "SeedMutationResult",
    "encoded_entries",
    "parsed_entries",
    "parsed_symmetrizer",
]
