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
MIN_KEMPNER_ARITY = 3
MAX_KEMPNER_ARITY = 999
_KEMPNER_BASE_PATTERN = r"^(?:[2-9]|[1-5][0-9]|6[0-4])(?![\s\S])"
_KEMPNER_ARITY_PATTERN = r"^(?:[3-9]|[1-9][0-9]{1,2})(?![\s\S])"

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
        json_schema_extra={"pattern": _KEMPNER_BASE_PATTERN},
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
        ge=MIN_KEMPNER_ARITY,
        le=MAX_KEMPNER_ARITY,
        description="Number of terms in the nontrivial progression, at least three.",
        json_schema_extra={"pattern": _KEMPNER_ARITY_PATTERN},
    )


class KempnerProgressionFree(StrictModel):
    """No nontrivial arithmetic progression exists in the digit family."""

    status: Literal["PROGRESSION_FREE"]


class KempnerContainsProgression(StrictModel):
    """One canonical shortest-padded witness of a fixed-arity progression."""

    status: Literal["CONTAINS_PROGRESSION"]
    indices: tuple[KempnerSmallInteger, ...] = Field(
        min_length=MIN_KEMPNER_ARITY,
        max_length=MAX_KEMPNER_ARITY,
    )
    values: tuple[KempnerInteger, ...] = Field(
        min_length=MIN_KEMPNER_ARITY,
        max_length=MAX_KEMPNER_ARITY,
    )
    first_term: KempnerInteger
    common_difference: KempnerInteger

    @model_validator(mode="after")
    def require_witness_shape(self) -> Self:
        if self.first_term < 1 or self.common_difference < 1:
            raise _validation_error(
                "witness_positive",
                "a progression witness must have positive first term and difference",
            )
        if len(self.values) < 3:
            raise _validation_error(
                "witness_shape",
                "a positive result must retain one ordered value for each source index",
            )
        expected = tuple(
            self.first_term + index * self.common_difference
            for index in range(len(self.values))
        )
        if self.values != expected or self.indices != tuple(range(len(self.values))):
            raise _validation_error(
                "witness_shape",
                "a positive result must retain one ordered value for each source index",
            )
        return self


class KempnerArithmeticProgressionResult(StrictModel):
    """An exact source-bound decision and, when present, one canonical witness."""

    digit_set: KempnerDigitSet
    arity: KempnerSmallInteger = Field(
        ge=MIN_KEMPNER_ARITY,
        le=MAX_KEMPNER_ARITY,
        description="Number of terms in the nontrivial progression, at least three.",
        json_schema_extra={"pattern": _KEMPNER_ARITY_PATTERN},
    )
    conclusion: Annotated[
        KempnerProgressionFree | KempnerContainsProgression,
        Field(discriminator="status"),
    ]

    @model_validator(mode="after")
    def require_arity_matches_witness(self) -> Self:
        if (
            isinstance(self.conclusion, KempnerContainsProgression)
            and len(self.conclusion.values) != self.arity
        ):
            raise _validation_error(
                "witness_shape",
                "a positive result must retain one ordered value for each source index",
            )
        return self

    @property
    def status(self) -> ProgressionStatus:
        return self.conclusion.status

    @property
    def indices(self) -> tuple[int, ...]:
        if isinstance(self.conclusion, KempnerContainsProgression):
            return self.conclusion.indices
        return ()

    @property
    def values(self) -> tuple[int, ...]:
        if isinstance(self.conclusion, KempnerContainsProgression):
            return self.conclusion.values
        return ()

    @property
    def first_term(self) -> int | None:
        if isinstance(self.conclusion, KempnerContainsProgression):
            return self.conclusion.first_term
        return None

    @property
    def common_difference(self) -> int | None:
        if isinstance(self.conclusion, KempnerContainsProgression):
            return self.conclusion.common_difference
        return None


__all__ = [
    "MAX_ALLOWED_DIGITS",
    "MAX_KEMPNER_BASE",
    "MAX_KEMPNER_INTEGER_DIGITS",
    "KempnerArithmeticProgressionRequest",
    "KempnerArithmeticProgressionResult",
    "KempnerContainsProgression",
    "KempnerDigitSet",
    "KempnerProgressionFree",
]
