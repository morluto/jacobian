"""Typed contracts for bounded powerful-number enumeration."""

from __future__ import annotations

import math
from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer

# Admission: enumerate squarefree b with b^3 <= cutoff, then a with a^2*b^3 <= cutoff.
# The total number of powerful integers up to X is at most 3*sqrt(X).
# We cap the cutoff by the complete family cardinality.
MAX_POWERFUL_ENUM_FAMILY_SIZE = 200_000
MAX_POWERFUL_ENUM_CUTOFF = ((MAX_POWERFUL_ENUM_FAMILY_SIZE - 1) // 3) ** 2
MAX_POWERFUL_ENUM_CUTOFF_DIGITS = len(str(MAX_POWERFUL_ENUM_CUTOFF))


class PowerfulEnumerateRequest(StrictModel):
    """Enumerate every powerful integer up to a positive upper bound."""

    cutoff: int = Field(
        gt=0,
        le=MAX_POWERFUL_ENUM_CUTOFF,
        description=(
            "Positive upper bound (inclusive). Every powerful integer in "
            "[1, cutoff] is returned exactly once in increasing order."
        ),
        examples=[100],
    )

    @model_validator(mode="after")
    def require_bounded_family(self) -> Self:
        estimate = 3 * math.isqrt(self.cutoff) + 1
        if estimate > MAX_POWERFUL_ENUM_FAMILY_SIZE:
            raise PydanticCustomError(
                "powerful_enumerate_family_exceeds_result_budget",
                "powerful family exceeds the result-size budget",
            )
        return self


class PowerfulEnumerateResult(StrictModel):
    """The complete ordered family of powerful integers up to the cutoff."""

    cutoff: int = Field(gt=0)
    count: int = Field(ge=0)
    family: tuple[CanonicalInteger, ...] = Field(default=())

    @model_validator(mode="after")
    def require_canonical_family(self) -> Self:
        if self.count != len(self.family):
            raise PydanticCustomError(
                "powerful_enumerate.count_mismatch",
                "count must equal the family length",
            )
        values = [int(v) for v in self.family]
        if any(v < 1 for v in values):
            raise PydanticCustomError(
                "powerful_enumerate.positive_only",
                "every powerful integer must be positive",
            )
        if values != sorted(values):
            raise PydanticCustomError(
                "powerful_enumerate.sorted",
                "powerful family must be sorted in increasing order",
            )
        if len(set(values)) != len(values):
            raise PydanticCustomError(
                "powerful_enumerate.unique",
                "powerful family must have no duplicates",
            )
        return self

    @classmethod
    def _from_kernel(
        cls, cutoff: int, raw_family: list[int]
    ) -> PowerfulEnumerateResult:
        family = tuple(format_canonical_integer(v) for v in sorted(raw_family))
        return cls.model_construct(
            cutoff=cutoff,
            count=len(family),
            family=family,
        )


__all__ = [
    "MAX_POWERFUL_ENUM_CUTOFF",
    "MAX_POWERFUL_ENUM_CUTOFF_DIGITS",
    "PowerfulEnumerateRequest",
    "PowerfulEnumerateResult",
]
