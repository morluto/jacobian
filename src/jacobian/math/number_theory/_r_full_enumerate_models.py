"""Typed contracts for bounded r-full integer enumeration."""

from __future__ import annotations

from bisect import bisect_right
from heapq import merge
from typing import NamedTuple, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError
from sympy import integer_nthroot
from sympy.ntheory.generate import primerange

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import (
    CanonicalizationError,
    CanonicalLimits,
    encode_strict_json,
    format_canonical_integer,
    parse_canonical_integer,
)

MAX_R_FULL_CUTOFF_DIGITS = 256
MAX_R_FULL_CUTOFF = 10**MAX_R_FULL_CUTOFF_DIGITS
MIN_R_FULL_EXPONENT = 2
MAX_R_FULL_EXPONENT = 850
MAX_R_FULL_FAMILY_SIZE = 200_000
MAX_R_FULL_RESULT_BYTES = CanonicalLimits().max_output_bytes
_MAX_PRIME_SEARCH_BOUND = 3_000_000


class RFullFamilyPlan(NamedTuple):
    """Request-scoped family plan shared by admission and result construction."""

    family: tuple[int, ...]
    exceeded: bool


def plan_r_full_family(minimum_exponent: int, cutoff: int) -> RFullFamilyPlan:
    """Return a conservative, exponent-sensitive family-size estimate.

    The admission pass counts the multiplicative family itself, stopping as
    soon as the result budget is exceeded.  This avoids the unsound fixed
    multiplier previously used for large exponents: prime-power products can
    be much more numerous than ``10 * cutoff**(1 / r)``.  A cutoff above the
    bounded prime-search range is rejected before asking SymPy to sieve a
    huge interval; the first 200,001 prime powers already exceed the result
    budget at that boundary.
    """
    prime_bound, _ = integer_nthroot(cutoff, minimum_exponent)
    if prime_bound > _MAX_PRIME_SEARCH_BOUND:
        return RFullFamilyPlan((), True)

    family_set: set[int] = {1}
    sorted_family = [1]
    for prime in primerange(2, int(prime_bound) + 1):
        powers: list[int] = []
        current = int(prime) ** minimum_exponent
        while current <= cutoff:
            powers.append(current)
            current *= int(prime)
        new_values: set[int] = set()
        for power in powers:
            limit = cutoff // power
            for member in sorted_family[: bisect_right(sorted_family, limit)]:
                new_values.add(member * power)
        fresh_values = sorted(value for value in new_values if value not in family_set)
        if fresh_values:
            family_set.update(fresh_values)
            sorted_family = list(merge(sorted_family, fresh_values))
            if len(sorted_family) > MAX_R_FULL_FAMILY_SIZE:
                return RFullFamilyPlan((), True)
    return RFullFamilyPlan(tuple(sorted_family), False)


def estimate_r_full_family_size(minimum_exponent: int, cutoff: int) -> int:
    """Return the admitted family size, or one past the family limit."""
    plan = plan_r_full_family(minimum_exponent, cutoff)
    return MAX_R_FULL_FAMILY_SIZE + 1 if plan.exceeded else len(plan.family)


def estimate_r_full_result_bytes(
    minimum_exponent: int, cutoff: int, family: tuple[int, ...]
) -> int:
    """Measure the canonical result payload for an admitted family."""
    payload = {
        "minimum_exponent": minimum_exponent,
        "cutoff": format_canonical_integer(cutoff),
        "count": len(family),
        "family": [format_canonical_integer(value) for value in family],
    }
    try:
        return len(
            encode_strict_json(
                payload,
                limits=CanonicalLimits(max_output_bytes=MAX_R_FULL_RESULT_BYTES),
            )
        )
    except CanonicalizationError:
        return MAX_R_FULL_RESULT_BYTES + 1


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
    "RFullFamilyPlan",
    "estimate_r_full_family_size",
    "estimate_r_full_result_bytes",
    "plan_r_full_family",
]
