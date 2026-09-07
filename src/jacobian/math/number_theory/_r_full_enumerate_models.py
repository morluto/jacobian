"""Typed contracts for bounded r-full integer enumeration."""

from __future__ import annotations

from bisect import bisect_right
from heapq import merge
from typing import Annotated, Literal, NamedTuple, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import DecimalIntegerEncoding
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer

MAX_R_FULL_CUTOFF_DIGITS = 32_768
MAX_R_FULL_CUTOFF = 10**MAX_R_FULL_CUTOFF_DIGITS
MIN_R_FULL_EXPONENT = 2
MAX_R_FULL_FAMILY_SIZE = 200_000
MAX_R_FULL_MERGE_WORK = 20_000_000
_MAX_PRIME_SEARCH_BOUND = 3_000_000


def max_r_full_exponent(cutoff: int) -> int:
    """Derive the maximum exponent from the cutoff and result budget.

    For a given cutoff, the largest meaningful exponent is one where the
    prime bound (cutoff^(1/r)) is at least 2, i.e. r <= log2(cutoff).
    Beyond that, the only r-full integer is 1 and the family is trivially
    [1, 2^r] when 2^r <= cutoff.  The family-size and merge-work budgets
    handle rejection during planning.
    """
    if cutoff < 2:
        return MIN_R_FULL_EXPONENT
    return max(MIN_R_FULL_EXPONENT, cutoff.bit_length())


class RFullFamilyPlan(NamedTuple):
    """Request-scoped family plan shared by admission and result construction."""

    family: tuple[int, ...]
    exceeded: bool
    reason: Literal["none", "family", "planning"] = "none"


def plan_r_full_family(
    minimum_exponent: int,
    cutoff: int,
) -> RFullFamilyPlan:
    """Return a conservative, exponent-sensitive family-size estimate.

    The admission pass counts the multiplicative family itself, stopping as
    soon as the result budget is exceeded.  This avoids the unsound fixed
    multiplier previously used for large exponents: prime-power products can
    be much more numerous than ``10 * cutoff**(1 / r)``.  A cutoff above the
    bounded prime-search range is rejected before asking SymPy to sieve a
    huge interval; the first 200,001 prime powers already exceed the result
    budget at that boundary.

    """
    # Keep the aggregate native namespace free of packaged backends.  The
    # planner is the first execution path that needs SymPy, so import it only
    # after request admission has selected this operation.
    from sympy import integer_nthroot
    from sympy.ntheory.generate import primerange

    prime_bound, _ = integer_nthroot(cutoff, minimum_exponent)
    if prime_bound > _MAX_PRIME_SEARCH_BOUND:
        return RFullFamilyPlan((), True)

    family_set: set[int] = {1}
    sorted_family = [1]
    merge_work = 0
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
                value = member * power
                if value in family_set or value in new_values:
                    continue
                if len(family_set) + len(new_values) >= MAX_R_FULL_FAMILY_SIZE:
                    return RFullFamilyPlan((), True, "family")
                new_values.add(value)
        fresh_values = sorted(value for value in new_values if value not in family_set)
        if fresh_values:
            merge_work += len(sorted_family) + len(fresh_values)
            if merge_work > MAX_R_FULL_MERGE_WORK:
                return RFullFamilyPlan((), True, "planning")
            family_set.update(fresh_values)
            sorted_family = list(merge(sorted_family, fresh_values))
            if len(sorted_family) > MAX_R_FULL_FAMILY_SIZE:
                return RFullFamilyPlan((), True, "family")
    return RFullFamilyPlan(tuple(sorted_family), False, "none")


def estimate_r_full_family_size(minimum_exponent: int, cutoff: int) -> int:
    """Return the admitted family size, or one past the family limit."""
    plan = plan_r_full_family(minimum_exponent, cutoff)
    return MAX_R_FULL_FAMILY_SIZE + 1 if plan.exceeded else len(plan.family)


class RFullEnumerateRequest(StrictModel):
    """Enumerate every r-full integer up to a positive upper bound."""

    minimum_exponent: int = Field(
        ge=MIN_R_FULL_EXPONENT,
        description=(
            "Minimum prime exponent r. An integer n is r-full when every "
            "prime factor appears to exponent at least r. The exponent "
            "ceiling is derived from the cutoff during admission."
        ),
        examples=[2],
    )
    cutoff: Annotated[
        int, DecimalIntegerEncoding(max_digits=MAX_R_FULL_CUTOFF_DIGITS)
    ] = Field(
        description="Positive upper bound (inclusive).",
        examples=["100"],
    )

    @model_validator(mode="after")
    def require_positive_cutoff(self) -> Self:
        cutoff = self.cutoff
        if cutoff <= 0 or cutoff > MAX_R_FULL_CUTOFF:
            raise PydanticCustomError(
                "r_full_enumerate.cutoff_bound",
                "cutoff must be a positive canonical integer within the admitted bound",
            )
        return self


class RFullEnumerateResult(StrictModel):
    """The complete ordered family of r-full integers up to the cutoff."""

    minimum_exponent: int = Field(ge=MIN_R_FULL_EXPONENT)
    cutoff: Annotated[int, DecimalIntegerEncoding(max_digits=MAX_R_FULL_CUTOFF_DIGITS)]
    count: int = Field(ge=0)
    family: tuple[
        Annotated[int, DecimalIntegerEncoding(max_digits=MAX_R_FULL_CUTOFF_DIGITS)], ...
    ] = Field(default=(), max_length=MAX_R_FULL_FAMILY_SIZE)

    @model_validator(mode="after")
    def require_canonical_family(self) -> Self:
        if self.count != len(self.family):
            raise PydanticCustomError(
                "r_full_enumerate.count_mismatch",
                "count must equal the family length",
            )
        cutoff = self.cutoff
        if cutoff <= 0 or cutoff > MAX_R_FULL_CUTOFF:
            raise PydanticCustomError(
                "r_full_enumerate.cutoff_bound",
                "cutoff must be a positive canonical integer within the admitted bound",
            )
        if any(
            len(format_canonical_integer(abs(value))) > MAX_R_FULL_CUTOFF_DIGITS
            for value in self.family
        ):
            raise PydanticCustomError(
                "r_full_enumerate.family_member_width",
                "family members exceed the admitted canonical width",
            )
        values = list(self.family)
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
        cutoff: int,
        raw_family: list[int],
    ) -> RFullEnumerateResult:
        family = tuple(sorted(raw_family))
        return cls.model_construct(
            minimum_exponent=minimum_exponent,
            cutoff=cutoff,
            count=len(family),
            family=family,
        )


__all__ = [
    "MAX_R_FULL_CUTOFF",
    "MAX_R_FULL_CUTOFF_DIGITS",
    "MIN_R_FULL_EXPONENT",
    "RFullEnumerateRequest",
    "RFullEnumerateResult",
    "RFullFamilyPlan",
    "estimate_r_full_family_size",
    "max_r_full_exponent",
    "plan_r_full_family",
]
