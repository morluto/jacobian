"""Typed contracts for the r-full enumeration operation."""

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel

MAX_R_FULL_SIEVE_BOUND = 1_000_000


class RFullEnumerationRequest(StrictModel):
    """Request to enumerate all r-full integers up to a bound."""

    bound: int = Field(ge=0)
    minimum_exponent: int = Field(ge=2)

    @model_validator(mode="after")
    def require_bounded_sieve(self) -> "RFullEnumerationRequest":
        if self.bound > MAX_R_FULL_SIEVE_BOUND and not _is_trivial_family(
            self.bound, self.minimum_exponent
        ):
            raise PydanticCustomError(
                "number_theory.r_full_sieve_bound",
                "r-full enumeration supports large bounds only when the family is (1,)",
            )
        return self


def _is_trivial_family(bound: int, minimum_exponent: int) -> bool:
    """Whether no prime power can occur below *bound*."""
    return bound >= 1 and minimum_exponent > bound.bit_length() - 1


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
