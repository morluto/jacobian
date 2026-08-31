"""Typed contracts for the Collatz-Wielandt quotient profile."""

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel


class CollatzWielandtRequest(StrictModel):
    """Request the Collatz-Wielandt quotient profile."""

    matrix: tuple[tuple[CanonicalRational, ...], ...] = Field(min_length=1)
    vector: tuple[CanonicalRational, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_nonnegative_square_problem(self) -> Self:
        n = len(self.vector)
        if len(self.matrix) != n or any(len(row) != n for row in self.matrix):
            raise PydanticCustomError(
                "collatz_wielandt.square_matrix",
                "matrix must be square and aligned with the vector",
            )
        if any(
            value.num.startswith("-") and value.num != "0"
            for row in self.matrix
            for value in row
        ):
            raise PydanticCustomError(
                "collatz_wielandt.nonnegative_matrix",
                "Collatz-Wielandt requires a nonnegative matrix",
            )
        if any(value.num.startswith("-") or value.num == "0" for value in self.vector):
            raise PydanticCustomError(
                "collatz_wielandt.positive_vector",
                "Collatz-Wielandt requires a strictly positive vector",
            )
        return self


class CollatzWielandtResult(StrictModel):
    """The Collatz-Wielandt quotient profile."""

    matrix: tuple[tuple[CanonicalRational, ...], ...]
    vector: tuple[CanonicalRational, ...]
    quotients: tuple[CanonicalRational, ...]
    max_quotient: CanonicalRational


__all__ = [
    "CollatzWielandtRequest",
    "CollatzWielandtResult",
]
