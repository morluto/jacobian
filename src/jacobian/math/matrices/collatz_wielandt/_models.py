"""Typed contracts for the Collatz-Wielandt quotient profile."""

from pydantic import Field

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.matrices.values import RationalMatrix


class CollatzWielandtRequest(StrictModel):
    """Request the Collatz-Wielandt quotient profile."""

    matrix: RationalMatrix
    vector: tuple[CanonicalRational, ...] = Field(min_length=1)


class CollatzWielandtResult(StrictModel):
    """The Collatz-Wielandt quotient profile."""

    matrix: RationalMatrix
    vector: tuple[CanonicalRational, ...]
    quotients: tuple[CanonicalRational, ...]
    max_quotient: CanonicalRational


__all__ = [
    "CollatzWielandtRequest",
    "CollatzWielandtResult",
]
