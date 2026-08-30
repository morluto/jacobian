"""Typed contracts for bounded r-full integer enumeration."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError
from sympy import integer_nthroot

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer, parse_canonical_integer

MAX_R_FULL_CUTOFF_DIGITS = 256
MAX_R_FULL_CUTOFF = 10**MAX_R_FULL_CUTOFF_DIGITS
MIN_R_FULL_EXPONENT = 2
MAX_R_FULL_EXPONENT = 64
MAX_R_FULL_FAMILY_SIZE = 200_000
MAX_R_FULL_RESULT_BYTES = 3_000_000


def estimate_r_full_family_size(minimum_exponent: int, cutoff: int) -> int:
    """Return a conservative estimate from the prime-power bound."""
    prime_bound, _ = integer_nthroot(cutoff, minimum_exponent)
    return max(1, 10 * int(prime_bound))


class RFullEnumerateRequest(StrictModel):
    """Enumerate every r-full integer up to a positive upper bound."""

    minimum_exponent: int = Field(
        ge=MIN_R_FULL_EXPONENT,
        le=MAX_R_FULL_EXPONENT,
        description=(
            "Minimum prime exponent r. An integer n is r-full when every "
            "prime factor appears to exponent at least r."
        ),
        examples=[2],
    )
    cutoff: CanonicalInteger = Field(
        max_length=MAX_R_FULL_CUTOFF_DIGITS + 1,
        description="Positive upper bound (inclusive).",
        examples=["100"],
    )

    @model_validator(mode="after")
    def require_positive_cutoff(self) -> Self:
        cutoff = parse_canonical_integer(self.cutoff)
        if cutoff <= 0 or cutoff > MAX_R_FULL_CUTOFF:
            raise PydanticCustomError(
                "r_full_enumerate.cutoff_bound",
                "cutoff must be a positive canonical integer within the admitted bound",
            )
        estimate = estimate_r_full_family_size(self.minimum_exponent, cutoff)
        if estimate > MAX_R_FULL_FAMILY_SIZE:
            raise PydanticCustomError(
                "r_full_enumerate.family_budget",
                "r-full family exceeds the result-size budget",
            )
        if estimate * (len(self.cutoff) + 3) > MAX_R_FULL_RESULT_BYTES:
            raise PydanticCustomError(
                "r_full_enumerate.transport_budget",
                "r-full family exceeds the serialized-byte budget",
            )
        return self


class RFullEnumerateResult(StrictModel):
    """The complete ordered family of r-full integers up to the cutoff."""

    minimum_exponent: int = Field(ge=MIN_R_FULL_EXPONENT, le=MAX_R_FULL_EXPONENT)
    cutoff: CanonicalInteger = Field(max_length=MAX_R_FULL_CUTOFF_DIGITS + 1)
    count: int = Field(ge=0)
    family: tuple[CanonicalInteger, ...] = Field(
        default=(), max_length=MAX_R_FULL_FAMILY_SIZE
    )

    @model_validator(mode="after")
    def require_canonical_family(self) -> Self:
        if self.count != len(self.family):
            raise PydanticCustomError(
                "r_full_enumerate.count_mismatch",
                "count must equal the family length",
            )
        cutoff = parse_canonical_integer(self.cutoff)
        if cutoff <= 0 or cutoff > MAX_R_FULL_CUTOFF:
            raise PydanticCustomError(
                "r_full_enumerate.cutoff_bound",
                "cutoff must be a positive canonical integer within the admitted bound",
            )
        estimate = estimate_r_full_family_size(self.minimum_exponent, cutoff)
        if estimate > MAX_R_FULL_FAMILY_SIZE:
            raise PydanticCustomError(
                "r_full_enumerate.family_budget",
                "r-full family exceeds the result-size budget",
            )
        if estimate * (len(self.cutoff) + 3) > MAX_R_FULL_RESULT_BYTES:
            raise PydanticCustomError(
                "r_full_enumerate.transport_budget",
                "r-full family exceeds the serialized-byte budget",
            )
        if any(len(value) > MAX_R_FULL_CUTOFF_DIGITS + 1 for value in self.family):
            raise PydanticCustomError(
                "r_full_enumerate.family_member_width",
                "family members exceed the admitted canonical width",
            )
        values = [parse_canonical_integer(v) for v in self.family]
        if any(v < 1 for v in values):
            raise PydanticCustomError(
                "r_full_enumerate.positive_only",
                "every r-full integer must be positive",
            )
        if cutoff >= 1 and (not values or values[0] != 1):
            raise PydanticCustomError(
                "r_full_enumerate.missing_one",
                "a complete r-full family must begin with 1",
            )
        if any(v > cutoff for v in values):
            raise PydanticCustomError(
                "r_full_enumerate.family_within_cutoff",
                "every family member must not exceed cutoff",
            )
        if values != sorted(values):
            raise PydanticCustomError(
                "r_full_enumerate.sorted",
                "r-full family must be sorted in increasing order",
            )
        if len(set(values)) != len(values):
            raise PydanticCustomError(
                "r_full_enumerate.unique",
                "r-full family must have no duplicates",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        minimum_exponent: int,
        cutoff: str,
        raw_family: list[int],
    ) -> RFullEnumerateResult:
        family = tuple(format_canonical_integer(v) for v in sorted(raw_family))
        return cls.model_construct(
            minimum_exponent=minimum_exponent,
            cutoff=cutoff,
            count=len(family),
            family=family,
        )


__all__ = [
    "MAX_R_FULL_CUTOFF",
    "MAX_R_FULL_CUTOFF_DIGITS",
    "MAX_R_FULL_EXPONENT",
    "MIN_R_FULL_EXPONENT",
    "RFullEnumerateRequest",
    "RFullEnumerateResult",
    "estimate_r_full_family_size",
]
