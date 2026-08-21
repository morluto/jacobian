"""Exact finite-dimensional linear algebra over an explicit prime field."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import ConfigDict, StrictInt
from pydantic.dataclasses import dataclass

_MAX_DIMENSION = 256

__all__ = [
    "PrimeFieldMatrix",
    "column_basis",
    "nullspace",
    "quotient_basis",
    "rank",
    "rref",
]


@dataclass(config=ConfigDict(extra="forbid"), frozen=True, slots=True)
class PrimeFieldMatrix:
    """An immutable matrix with an exact prime and explicit empty shape."""

    prime: StrictInt
    entries: tuple[tuple[StrictInt, ...], ...]
    columns: StrictInt

    def __post_init__(self) -> None:
        if type(self.prime) is not int or self.prime < 2:
            raise ValueError("prime must be a prime integer")
        if type(self.columns) is not int or self.columns < 0:
            raise ValueError("columns must be a nonnegative integer")
        if self.columns > _MAX_DIMENSION or len(self.entries) > _MAX_DIMENSION:
            raise ValueError("matrix exceeds the supported dimension bound")
        if any(len(row) != self.columns for row in self.entries):
            raise ValueError("every matrix row must match the declared column count")
        if any(
            type(value) is not int or not 0 <= value < self.prime
            for row in self.entries
            for value in row
        ):
            raise ValueError("matrix entries must be canonical prime-field residues")
        from sympy import isprime

        if not isprime(self.prime):
            raise ValueError("prime must be a prime integer")


def _domain_matrix(matrix: PrimeFieldMatrix) -> Any:
    import sympy
    from sympy.polys.matrices import DomainMatrix

    entries = [list(row) for row in matrix.entries]
    return DomainMatrix(
        entries,
        (len(matrix.entries), matrix.columns),
        sympy.GF(matrix.prime),
    )


def rref(
    matrix: PrimeFieldMatrix,
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Return reduced rows and pivot columns over the bound prime field."""

    row_count = len(matrix.entries)
    if row_count == 0 or matrix.columns == 0:
        return tuple((0,) * matrix.columns for _ in matrix.entries), ()
    reduced_domain, pivot_columns = _domain_matrix(matrix).rref()
    reduced = reduced_domain.to_Matrix()
    return (
        tuple(
            tuple(
                int(reduced[row, column]) % matrix.prime
                for column in range(matrix.columns)
            )
            for row in range(row_count)
        ),
        tuple(int(pivot) for pivot in pivot_columns),
    )


def rank(matrix: PrimeFieldMatrix) -> int:
    """Return matrix rank over the bound prime field."""

    if not matrix.entries or matrix.columns == 0:
        return 0
    return int(_domain_matrix(matrix).rank())


def nullspace(matrix: PrimeFieldMatrix) -> tuple[tuple[int, ...], ...]:
    """Return a deterministic basis of the right nullspace."""

    if matrix.columns == 0:
        return ()
    domain = _domain_matrix(matrix).nullspace(divide_last=True)
    return tuple(
        tuple(int(value) % matrix.prime for value in row) for row in domain.to_list()
    )


def column_basis(matrix: PrimeFieldMatrix) -> tuple[tuple[int, ...], ...]:
    """Return the first independent columns in source order."""

    if matrix.columns == 0 or not matrix.entries:
        return ()
    _, pivots = rref(matrix)
    return tuple(
        tuple(row[pivot] % matrix.prime for row in matrix.entries) for pivot in pivots
    )


def quotient_basis(
    cycles: Sequence[Sequence[int]],
    boundaries: Sequence[Sequence[int]],
    *,
    prime: int,
) -> tuple[tuple[int, ...], ...]:
    """Extend a boundary basis by deterministic representatives of a quotient."""

    # Validate the prime even for the empty quotient.
    dimension = len(cycles[0]) if cycles else (len(boundaries[0]) if boundaries else 0)
    if any(len(vector) != dimension for vector in (*cycles, *boundaries)):
        raise ValueError("basis vector has the wrong dimension")
    PrimeFieldMatrix(prime=prime, entries=(), columns=dimension)
    if dimension == 0 or not cycles:
        return ()
    columns = tuple(
        tuple(int(value) % prime for value in vector) for vector in boundaries
    ) + tuple(tuple(int(value) % prime for value in vector) for vector in cycles)
    stacked = PrimeFieldMatrix(
        prime=prime,
        entries=tuple(
            tuple(vector[row] for vector in columns) for row in range(dimension)
        ),
        columns=len(columns),
    )
    _, pivots = rref(stacked)
    boundary_count = len(boundaries)
    return tuple(
        tuple(cycles[pivot - boundary_count])
        for pivot in pivots
        if pivot >= boundary_count
    )
