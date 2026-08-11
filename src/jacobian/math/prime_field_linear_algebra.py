"""Exact finite-dimensional linear algebra over an explicit prime field."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "PrimeFieldMatrix",
    "column_basis",
    "nullspace",
    "quotient_basis",
    "rank",
    "rref",
]


@dataclass(frozen=True, slots=True)
class PrimeFieldMatrix:
    """An immutable matrix with an exact prime and explicit empty shape."""

    prime: int
    entries: tuple[tuple[int, ...], ...]
    columns: int

    def __post_init__(self) -> None:
        from sympy import isprime

        if type(self.prime) is not int or not isprime(self.prime):
            raise ValueError("prime must be a prime integer")
        if type(self.columns) is not int or self.columns < 0:
            raise ValueError("columns must be a nonnegative integer")
        if any(len(row) != self.columns for row in self.entries):
            raise ValueError("every matrix row must match the declared column count")


def _domain_matrix(matrix: PrimeFieldMatrix) -> Any:
    import sympy
    from sympy.polys.matrices import DomainMatrix

    entries = [[value % matrix.prime for value in row] for row in matrix.entries]
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

    reduced, pivots = rref(matrix)
    free_columns = tuple(
        column for column in range(matrix.columns) if column not in pivots
    )
    basis: list[tuple[int, ...]] = []
    for free in free_columns:
        vector = [0] * matrix.columns
        vector[free] = 1
        for row, pivot in enumerate(pivots):
            vector[pivot] = -reduced[row][free] % matrix.prime
        basis.append(tuple(vector))
    return tuple(basis)


class _IncrementalVectorBasis:
    def __init__(self, *, dimension: int, prime: int) -> None:
        self._prime = prime
        self._rows: dict[int, list[int]] = {}
        self._dimension = dimension

    def add(self, vector: Sequence[int]) -> bool:
        reduced = [value % self._prime for value in vector]
        if len(reduced) != self._dimension:
            raise ValueError("basis vector has the wrong dimension")
        for existing_pivot, row in self._rows.items():
            factor = reduced[existing_pivot]
            if factor:
                reduced = [
                    (value - factor * basis_value) % self._prime
                    for value, basis_value in zip(reduced, row, strict=True)
                ]
        new_pivot = next(
            (index for index, value in enumerate(reduced) if value),
            None,
        )
        if new_pivot is None:
            return False
        inverse = pow(reduced[new_pivot], -1, self._prime)
        reduced = [value * inverse % self._prime for value in reduced]
        for existing_pivot, row in tuple(self._rows.items()):
            factor = row[new_pivot]
            if factor:
                self._rows[existing_pivot] = [
                    (value - factor * basis_value) % self._prime
                    for value, basis_value in zip(row, reduced, strict=True)
                ]
        self._rows[new_pivot] = reduced
        self._rows = dict(sorted(self._rows.items()))
        return True


def column_basis(matrix: PrimeFieldMatrix) -> tuple[tuple[int, ...], ...]:
    """Return the first independent columns in source order."""

    if matrix.columns == 0:
        return ()
    selected: list[tuple[int, ...]] = []
    basis = _IncrementalVectorBasis(
        dimension=len(matrix.entries),
        prime=matrix.prime,
    )
    for column in range(matrix.columns):
        vector = tuple(row[column] % matrix.prime for row in matrix.entries)
        if basis.add(vector):
            selected.append(vector)
    return tuple(selected)


def quotient_basis(
    cycles: Sequence[Sequence[int]],
    boundaries: Sequence[Sequence[int]],
    *,
    prime: int,
) -> tuple[tuple[int, ...], ...]:
    """Extend a boundary basis by deterministic representatives of a quotient."""

    # Validate the prime even for the empty quotient.
    dimension = len(cycles[0]) if cycles else (len(boundaries[0]) if boundaries else 0)
    PrimeFieldMatrix(prime=prime, entries=(), columns=dimension)
    basis = _IncrementalVectorBasis(dimension=dimension, prime=prime)
    for boundary in boundaries:
        basis.add(boundary)
    quotient: list[tuple[int, ...]] = []
    for cycle in cycles:
        vector = tuple(cycle)
        if basis.add(vector):
            quotient.append(vector)
    return tuple(quotient)
