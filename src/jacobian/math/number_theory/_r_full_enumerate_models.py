"""Typed contracts for bounded r-full integer enumeration."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer

MAX_R_FULL_CUTOFF_DIGITS = 18
MAX_R_FULL_CUTOFF = 10**MAX_R_FULL_CUTOFF_DIGITS
MIN_R_FULL_EXPONENT = 2
MAX_R_FULL_EXPONENT = 64
MAX_R_FULL_FAMILY_SIZE = 200_000
MAX_R_FULL_RESULT_BYTES = 3_000_000


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
    cutoff: int = Field(
        gt=0,
        le=MAX_R_FULL_CUTOFF,
        description="Positive upper bound (inclusive).",
        examples=[100],
    )


class RFullEnumerateResult(StrictModel):
    """The complete ordered family of r-full integers up to the cutoff."""

    minimum_exponent: int = Field(ge=MIN_R_FULL_EXPONENT, le=MAX_R_FULL_EXPONENT)
    cutoff: int = Field(gt=0)
    count: int = Field(ge=0)
    family: tuple[CanonicalInteger, ...] = Field(default=())

    @model_validator(mode="after")
    def require_canonical_family(self) -> Self:
        if self.count != len(self.family):
            raise PydanticCustomError(
                "r_full_enumerate.count_mismatch",
                "count must equal the family length",
            )
        values = [int(v) for v in self.family]
        if any(v < 1 for v in values):
            raise PydanticCustomError(
                "r_full_enumerate.positive_only",
                "every r-full integer must be positive",
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
]
