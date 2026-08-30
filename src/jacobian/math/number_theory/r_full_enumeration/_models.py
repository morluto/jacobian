"""Typed contracts for the r-full enumeration operation."""

from pydantic import Field

from jacobian._models import StrictModel

MAX_R_FULL_SIEVE_BOUND = 1_000_000


class RFullEnumerationRequest(StrictModel):
    """Request to enumerate all r-full integers up to a bound."""

    bound: int = Field(ge=0, le=MAX_R_FULL_SIEVE_BOUND)
    minimum_exponent: int = Field(ge=2)


class RFullEnumerationResult(StrictModel):
    """The complete bounded r-full family."""

    bound: int
    minimum_exponent: int
    values: tuple[int, ...]
    count: int


__all__ = [
    "MAX_R_FULL_SIEVE_BOUND",
    "RFullEnumerationRequest",
    "RFullEnumerationResult",
]
