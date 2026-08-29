"""Typed contracts for the r-full enumeration operation."""

from jacobian._models import StrictModel


class RFullEnumerationRequest(StrictModel):
    """Request to enumerate all r-full integers up to a bound."""

    bound: int
    minimum_exponent: int


class RFullEnumerationResult(StrictModel):
    """The complete bounded r-full family."""

    bound: int
    minimum_exponent: int
    values: tuple[int, ...]
    count: int


__all__ = [
    "RFullEnumerationRequest",
    "RFullEnumerationResult",
]
