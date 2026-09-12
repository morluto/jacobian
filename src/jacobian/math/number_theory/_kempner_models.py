"""Typed contracts for digit-restricted integer families."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import DecimalIntegerEncoding
from jacobian._models import StrictModel

MAX_KEMPNER_BASE = 64
MAX_KEMPNER_INTEGER_DIGITS = 8_192
MAX_ALLOWED_DIGITS = MAX_KEMPNER_BASE - 1

KempnerSmallInteger = Annotated[int, DecimalIntegerEncoding(max_digits=3)]
KempnerInteger = Annotated[
    int, DecimalIntegerEncoding(max_digits=MAX_KEMPNER_INTEGER_DIGITS)
]


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"number_theory.kempner.{reason}", message)


class KempnerDigitSet(StrictModel):
    """A canonical proper nonempty subset of one base's digit alphabet.

    Positive members are integers whose ordinary base-``base`` expansion has
    no leading zero and uses only ``allowed_digits``.  A zero digit in the
    alphabet is an ordinary significant digit; leading zero padding is never
    tested for membership.
    """

    base: KempnerSmallInteger = Field(
        ge=2,
        le=MAX_KEMPNER_BASE,
        description=f"Integer base in [2, {MAX_KEMPNER_BASE}].",
    )
    allowed_digits: tuple[KempnerSmallInteger, ...] = Field(
        min_length=1,
        max_length=MAX_ALLOWED_DIGITS,
        description="Strictly increasing proper subset of the base digit alphabet.",
    )

    @model_validator(mode="after")
    def require_canonical_digit_subset(self) -> Self:
        digits = self.allowed_digits
        if digits != tuple(sorted(set(digits))):
            raise _validation_error(
                "digit_subset_not_canonical",
                "allowed_digits must be strictly increasing with no duplicates",
            )
        if any(digit < 0 or digit >= self.base for digit in digits):
            raise _validation_error(
                "digit_out_of_range",
                "allowed_digits must lie in the base digit alphabet",
            )
        if len(digits) == self.base:
            raise _validation_error(
                "proper_subset_required",
                "allowed_digits must be a proper subset of the base alphabet",
            )
        return self


ProgressionStatus = Literal["PROGRESSION_FREE", "CONTAINS_PROGRESSION"]


class KempnerArithmeticProgressionRequest(StrictModel):
    """One exact fixed-arity progression decision request."""

    digit_set: KempnerDigitSet
    arity: KempnerSmallInteger = Field(
        ge=3,
        description="Number of terms in the nontrivial progression, at least three.",
    )


class KempnerArithmeticProgressionResult(StrictModel):
    """An exact source-bound decision and, when present, one canonical witness."""

    digit_set: KempnerDigitSet
    arity: KempnerSmallInteger
    status: ProgressionStatus
    indices: tuple[KempnerSmallInteger, ...]
    values: tuple[KempnerInteger, ...]
    first_term: KempnerInteger | None = None
    common_difference: KempnerInteger | None = None

    @model_validator(mode="after")
    def require_witness_shape(self) -> Self:
        expected_indices = tuple(range(self.arity))
        if self.status == "PROGRESSION_FREE":
            if (
                self.indices
                or self.values
                or self.first_term is not None
                or self.common_difference is not None
            ):
                raise _validation_error(
                    "free_result_witness",
                    "a progression-free result cannot contain a witness",
                )
            return self
        if self.indices != expected_indices or len(self.values) != self.arity:
            raise _validation_error(
                "witness_shape",
                "a positive result must retain one ordered value for each source index",
            )
        if self.first_term is None or self.common_difference is None:
            raise _validation_error(
                "witness_parameters",
                "a positive result must retain its first term and common difference",
            )
        if self.first_term < 1 or self.common_difference < 1:
            raise _validation_error(
                "witness_positive",
                "a progression witness must have positive first term and difference",
            )
        return self


__all__ = [
    "MAX_ALLOWED_DIGITS",
    "MAX_KEMPNER_BASE",
    "MAX_KEMPNER_INTEGER_DIGITS",
    "KempnerArithmeticProgressionRequest",
    "KempnerArithmeticProgressionResult",
    "KempnerDigitSet",
]
