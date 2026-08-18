"""Typed wire contracts for root system operations."""

from __future__ import annotations

from typing import Self

from pydantic import model_validator

from jacobian._models import StrictModel

MAX_RANK = 8


class CartanMatrixRequest(StrictModel):
    """A bounded finite-type Cartan matrix."""

    matrix: tuple[tuple[int, ...], ...]

    @model_validator(mode="after")
    def require_valid_cartan(self) -> Self:
        n = len(self.matrix)
        if n < 1 or n > MAX_RANK:
            raise ValueError(f"rank must be between 1 and {MAX_RANK}")
        for row in self.matrix:
            if len(row) != n:
                raise ValueError("Cartan matrix must be square")
        for i in range(n):
            if self.matrix[i][i] != 2:
                raise ValueError("diagonal entries must be 2")
        self._check_off_diagonal(n)
        return self

    def _check_off_diagonal(self, n: int) -> None:
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                aij = self.matrix[i][j]
                aji = self.matrix[j][i]
                if aij > 0:
                    raise ValueError("off-diagonal entries must be non-positive")
                if aij * aji not in (0, 1, 2, 3):
                    raise ValueError("off-diagonal product must be 0, 1, 2, or 3")


class PositiveRootsResult(StrictModel):
    """The positive roots of a root system."""

    rank: int
    positive_roots: tuple[tuple[int, ...], ...]
    num_positive_roots: int


class SimpleReflectionResult(StrictModel):
    """Result of applying a simple reflection to a vector."""

    vector: tuple[int, ...]
    simple_index: int
    reflected_vector: tuple[int, ...]


class RootSystemDataResult(StrictModel):
    """Complete root system data from a Cartan matrix."""

    rank: int
    cartan_matrix: tuple[tuple[int, ...], ...]
    positive_roots: tuple[tuple[int, ...], ...]
    negative_roots: tuple[tuple[int, ...], ...]
    simple_roots: tuple[tuple[int, ...], ...]
    highest_root: tuple[int, ...] | None
    num_positive_roots: int
    coxeter_number: int
