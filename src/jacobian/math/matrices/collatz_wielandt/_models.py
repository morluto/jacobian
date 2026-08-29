"""Typed contracts for the Collatz-Wielandt quotient profile."""

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel


class CollatzWielandtRequest(StrictModel):
    """Request the Collatz-Wielandt quotient profile."""

    matrix: tuple[tuple[CanonicalRational, ...], ...]
    vector: tuple[CanonicalRational, ...]


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
