"""Certified finite quadratic-surd and simultaneous-approximation values.

These values keep exact integer structure and carry a certified rational
enclosure whenever the underlying real number is irrational.  Precision is
requested as a binary scale ``2**-scale_bits`` so every refinement step is
deterministic integer arithmetic rather than a floating-point round.
"""

from __future__ import annotations

from math import isqrt
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import ExactInteger, require_bounded_rational
from jacobian._models import StrictModel
from jacobian.math.analysis.intervals import ClosedRationalInterval

MAX_SURD_RADICAND = 1_000_000
MAX_SURD_SCALE_BITS = 4_096
MAX_RANGE_LENGTH = 4_096
MAX_SIMULTANEOUS_RADICANDS = 8
# A product denominator has at most 8 * 4096 binary places; multiplying by
# a 4096-bit integer keeps both components below 16384 decimal digits.
MAX_SURD_MULTIPLIER_BITS = 4_096
MAX_ENCLOSURE_COMPONENT_DIGITS = 16_384


def _validation_error(code: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(code, message)


def is_surd_radicand(radicand: int) -> bool:
    """A bounded positive nonsquare radicand has an irrational square root."""

    if radicand < 2 or radicand > MAX_SURD_RADICAND:
        return False
    root = isqrt(radicand)
    return root * root != radicand


class PositiveQuadraticIrrational(StrictModel):
    """The positive square root of one bounded nonsquare integer radicand."""

    radicand: StrictInt = Field(ge=2, le=MAX_SURD_RADICAND)

    @model_validator(mode="after")
    def require_non_square_radicand(self) -> Self:
        if not is_surd_radicand(self.radicand):
            raise _validation_error(
                "diophantine.surd_radicand_must_not_be_square",
                "a quadratic irrational requires a nonsquare radicand",
            )
        return self


def bound_enclosure(
    interval: ClosedRationalInterval, *, label: str
) -> ClosedRationalInterval:
    """Reject an enclosure whose canonical components exceed the digit bound."""

    require_bounded_rational(
        interval.lower, max_digits=MAX_ENCLOSURE_COMPONENT_DIGITS, label=label
    )
    require_bounded_rational(
        interval.upper, max_digits=MAX_ENCLOSURE_COMPONENT_DIGITS, label=label
    )
    return interval


class ScaledFloorRequest(StrictModel):
    """Exact floor/ceiling of ``n * sqrt(d)`` for positive ``n`` and nonsquare ``d``."""

    multiplier: ExactInteger = Field(
        ge=1,
        description="Positive exact multiplier; computation admits at most 4096 bits.",
    )
    radicand: StrictInt = Field(ge=2, le=MAX_SURD_RADICAND)

    @model_validator(mode="after")
    def require_non_square_radicand(self) -> Self:
        if not is_surd_radicand(self.radicand):
            raise _validation_error(
                "diophantine.surd_radicand_must_not_be_square",
                "a quadratic irrational requires a nonsquare radicand",
            )
        return self


class ScaledFloorValue(StrictModel):
    """One exact floor/ceiling row derived from integer squares only."""

    multiplier: ExactInteger = Field(
        ge=1,
        description="Positive exact multiplier; computation admits at most 4096 bits.",
    )
    radicand: StrictInt = Field(ge=2, le=MAX_SURD_RADICAND)
    floor: ExactInteger
    ceiling: ExactInteger
    square_lower: ExactInteger
    square_upper: ExactInteger

    @model_validator(mode="after")
    def require_exact_square_bracket(self) -> Self:
        if self.ceiling != self.floor + 1:
            raise _validation_error(
                "diophantine.scaled_floor_not_unit_bracket",
                "an irrational scaled floor must have ceiling equal to floor + 1",
            )
        if not (
            self.floor >= 0
            and self.square_lower == self.floor * self.floor
            and self.square_upper == self.ceiling * self.ceiling
        ):
            raise _validation_error(
                "diophantine.scaled_floor_square_mismatch",
                "square_lower/square_upper must be the exact endpoint squares",
            )
        return self


class ScaledFloorResult(StrictModel):
    """A bounded carrier of exact scaled floors over an ordered radicand axis."""

    rows: tuple[ScaledFloorValue, ...] = Field(
        min_length=1, max_length=MAX_RANGE_LENGTH
    )


class NearestIntegerDistanceRequest(StrictModel):
    """Certify ``||n sqrt(d)||`` at a requested binary precision."""

    multiplier: ExactInteger = Field(
        ge=1,
        description="Positive exact multiplier; computation admits at most 4096 bits.",
    )
    radicand: StrictInt = Field(ge=2, le=MAX_SURD_RADICAND)
    scale_bits: StrictInt = Field(ge=1, le=MAX_SURD_SCALE_BITS)

    @model_validator(mode="after")
    def require_non_square_radicand(self) -> Self:
        if not is_surd_radicand(self.radicand):
            raise _validation_error(
                "diophantine.surd_radicand_must_not_be_square",
                "a quadratic irrational requires a nonsquare radicand",
            )
        return self


class NearestIntegerDistanceValue(StrictModel):
    """Exact nearest-integer data plus a certified distance enclosure."""

    multiplier: ExactInteger = Field(
        ge=1,
        description="Positive exact multiplier; computation admits at most 4096 bits.",
    )
    radicand: StrictInt = Field(ge=2, le=MAX_SURD_RADICAND)
    floor: ExactInteger
    ceiling: ExactInteger
    nearest_integer: ExactInteger
    side: Literal["FLOOR", "CEILING"]
    scale_bits: StrictInt = Field(ge=1, le=MAX_SURD_SCALE_BITS)
    distance_enclosure: ClosedRationalInterval
    distance_upper_scaled: ExactInteger

    @model_validator(mode="after")
    def require_nearest_branch(self) -> Self:
        if self.ceiling != self.floor + 1 or self.floor < 0:
            raise _validation_error(
                "diophantine.nearest_integer_not_unit_bracket",
                "nearest-integer distance requires a nonnegative unit bracket",
            )
        expected_nearest = self.floor if self.side == "FLOOR" else self.ceiling
        if self.nearest_integer != expected_nearest:
            raise _validation_error(
                "diophantine.nearest_integer_branch_mismatch",
                "nearest_integer must be the declared bracket endpoint",
            )
        if self.distance_upper_scaled < 0:
            raise _validation_error(
                "diophantine.nearest_integer_negative_scale",
                "distance_upper_scaled is a nonnegative integer numerator",
            )
        return self


class SimultaneousProductRequest(StrictModel):
    """Certify ``n * prod_i ||n sqrt(d_i)||`` over an ordered radicand axis."""

    multiplier: ExactInteger = Field(
        ge=1,
        description="Positive exact multiplier; computation admits at most 4096 bits.",
    )
    radicands: tuple[StrictInt, ...] = Field(
        min_length=1, max_length=MAX_SIMULTANEOUS_RADICANDS
    )
    scale_bits: StrictInt = Field(ge=1, le=MAX_SURD_SCALE_BITS)

    @model_validator(mode="after")
    def require_distinct_non_square_radicands(self) -> Self:
        if any(not is_surd_radicand(radicand) for radicand in self.radicands):
            raise _validation_error(
                "diophantine.surd_radicand_must_not_be_square",
                "every radicand must be a nonsquare integer in the admitted range",
            )
        if len(set(self.radicands)) != len(self.radicands):
            raise _validation_error(
                "diophantine.surd_radicands_must_be_distinct",
                "an ordered radicand axis requires distinct radicands",
            )
        return self


class SimultaneousProductResult(StrictModel):
    """Per-factor nearest-integer rows and the certified product enclosure."""

    multiplier: ExactInteger = Field(
        ge=1,
        description="Positive exact multiplier; computation admits at most 4096 bits.",
    )
    radicands: tuple[StrictInt, ...] = Field(
        min_length=1, max_length=MAX_SIMULTANEOUS_RADICANDS
    )
    scale_bits: StrictInt = Field(ge=1, le=MAX_SURD_SCALE_BITS)
    factors: tuple[NearestIntegerDistanceValue, ...] = Field(
        min_length=1, max_length=MAX_SIMULTANEOUS_RADICANDS
    )
    product_enclosure: ClosedRationalInterval

    @model_validator(mode="after")
    def require_factor_axis_alignment(self) -> Self:
        if tuple(factor.radicand for factor in self.factors) != self.radicands:
            raise _validation_error(
                "diophantine.product_factor_axis_mismatch",
                "factor rows must follow the declared radicand axis",
            )
        if any(factor.multiplier != self.multiplier for factor in self.factors):
            raise _validation_error(
                "diophantine.product_factor_multiplier_mismatch",
                "every factor row must share the request multiplier",
            )
        return self


class RangeProfileRequest(StrictModel):
    """A complete certified row for every ``1 <= n <= limit``."""

    radicands: tuple[StrictInt, ...] = Field(
        min_length=1, max_length=MAX_SIMULTANEOUS_RADICANDS
    )
    limit: StrictInt = Field(ge=1, le=MAX_RANGE_LENGTH)
    scale_bits: StrictInt = Field(ge=1, le=MAX_SURD_SCALE_BITS)

    @model_validator(mode="after")
    def require_distinct_non_square_radicands(self) -> Self:
        if any(not is_surd_radicand(radicand) for radicand in self.radicands):
            raise _validation_error(
                "diophantine.surd_radicand_must_not_be_square",
                "every radicand must be a nonsquare integer in the admitted range",
            )
        if len(set(self.radicands)) != len(self.radicands):
            raise _validation_error(
                "diophantine.surd_radicands_must_be_distinct",
                "an ordered radicand axis requires distinct radicands",
            )
        return self


class RangeProfileRow(StrictModel):
    """One complete certified row: factor enclosures and their product."""

    multiplier: ExactInteger = Field(
        ge=1,
        description="Positive exact multiplier; computation admits at most 4096 bits.",
    )
    factors: tuple[NearestIntegerDistanceValue, ...] = Field(
        min_length=1, max_length=MAX_SIMULTANEOUS_RADICANDS
    )
    product_enclosure: ClosedRationalInterval


class RangeProfileResult(StrictModel):
    """Every row of the declared range with no silent omission."""

    radicands: tuple[StrictInt, ...] = Field(
        min_length=1, max_length=MAX_SIMULTANEOUS_RADICANDS
    )
    limit: StrictInt = Field(ge=1, le=MAX_RANGE_LENGTH)
    scale_bits: StrictInt = Field(ge=1, le=MAX_SURD_SCALE_BITS)
    rows: tuple[RangeProfileRow, ...] = Field(max_length=MAX_RANGE_LENGTH)

    @model_validator(mode="after")
    def require_complete_range(self) -> Self:
        if len(self.rows) != self.limit:
            raise _validation_error(
                "diophantine.range_profile_incomplete",
                "the range profile must contain exactly one row per integer",
            )
        if tuple(row.multiplier for row in self.rows) != tuple(
            range(1, self.limit + 1)
        ):
            raise _validation_error(
                "diophantine.range_profile_multiplier_order",
                "range rows must be ordered by multiplier starting at one",
            )
        for row in self.rows:
            if tuple(factor.radicand for factor in row.factors) != self.radicands:
                raise _validation_error(
                    "diophantine.range_profile_factor_axis_mismatch",
                    "every row must follow the declared radicand axis",
                )
        return self


class RecordMinimaRequest(StrictModel):
    """Extract strict record minima from a certified simultaneous range."""

    radicands: tuple[StrictInt, ...] = Field(
        min_length=1, max_length=MAX_SIMULTANEOUS_RADICANDS
    )
    limit: StrictInt = Field(ge=1, le=MAX_RANGE_LENGTH)
    scale_bits: StrictInt = Field(ge=1, le=MAX_SURD_SCALE_BITS)

    @model_validator(mode="after")
    def require_distinct_non_square_radicands(self) -> Self:
        if any(not is_surd_radicand(radicand) for radicand in self.radicands):
            raise _validation_error(
                "diophantine.surd_radicand_must_not_be_square",
                "every radicand must be a nonsquare integer in the admitted range",
            )
        if len(set(self.radicands)) != len(self.radicands):
            raise _validation_error(
                "diophantine.surd_radicands_must_be_distinct",
                "an ordered radicand axis requires distinct radicands",
            )
        return self


class RecordMinimumValue(StrictModel):
    """One certified strict-record row."""

    multiplier: ExactInteger = Field(
        ge=1,
        description="Positive exact multiplier; computation admits at most 4096 bits.",
    )
    product_enclosure: ClosedRationalInterval
    incumbent_enclosure: ClosedRationalInterval


class RecordMinimaResult(StrictModel):
    """Certified strict record minima, or an explicit non-conclusion."""

    radicands: tuple[StrictInt, ...] = Field(
        min_length=1, max_length=MAX_SIMULTANEOUS_RADICANDS
    )
    limit: StrictInt = Field(ge=1, le=MAX_RANGE_LENGTH)
    scale_bits: StrictInt = Field(ge=1, le=MAX_SURD_SCALE_BITS)
    outcome: Literal["COMPLETE", "UNRESOLVED"]
    records: tuple[RecordMinimumValue, ...] = Field(max_length=MAX_RANGE_LENGTH)
    finite_argmin: StrictInt | None = None
    unresolved_multiplier: StrictInt | None = None
    unresolved_product_enclosure: ClosedRationalInterval | None = None
    unresolved_incumbent_enclosure: ClosedRationalInterval | None = None

    @model_validator(mode="after")
    def require_outcome_shape(self) -> Self:
        if self.outcome == "COMPLETE":
            if (
                self.unresolved_multiplier is not None
                or self.unresolved_product_enclosure is not None
                or self.unresolved_incumbent_enclosure is not None
            ):
                raise _validation_error(
                    "diophantine.complete_record_has_unresolved_fields",
                    "a complete record result cannot carry unresolved fields",
                )
            if not self.records:
                raise _validation_error(
                    "diophantine.complete_record_missing_first_row",
                    "a complete record sequence contains at least the first row",
                )
            if self.finite_argmin != self.records[-1].multiplier:
                raise _validation_error(
                    "diophantine.complete_record_argmin_mismatch",
                    "finite_argmin must be the last certified record multiplier",
                )
            for previous, current in zip(self.records, self.records[1:], strict=False):
                if current.multiplier <= previous.multiplier:
                    raise _validation_error(
                        "diophantine.complete_record_order",
                        "certified record multipliers must be strictly increasing",
                    )
                if not (
                    current.product_enclosure.upper.as_fraction()
                    < current.incumbent_enclosure.lower.as_fraction()
                ):
                    raise _validation_error(
                        "diophantine.complete_record_not_separated",
                        "every reported record must lie strictly below its incumbent",
                    )
        else:
            if self.finite_argmin is not None:
                raise _validation_error(
                    "diophantine.unresolved_record_has_argmin",
                    "an unresolved record search makes no argmin claim",
                )
            if (
                self.unresolved_multiplier is None
                or self.unresolved_product_enclosure is None
                or self.unresolved_incumbent_enclosure is None
            ):
                raise _validation_error(
                    "diophantine.unresolved_record_missing_evidence",
                    "an unresolved record search names the first overlapping row",
                )
            if (
                self.records
                and self.unresolved_multiplier <= self.records[-1].multiplier
            ):
                raise _validation_error(
                    "diophantine.unresolved_record_order",
                    "the unresolved multiplier must follow all certified records",
                )
            candidate = self.unresolved_product_enclosure
            incumbent = self.unresolved_incumbent_enclosure
            if (
                candidate.upper.as_fraction() < incumbent.lower.as_fraction()
                or incumbent.upper.as_fraction() <= candidate.lower.as_fraction()
            ):
                raise _validation_error(
                    "diophantine.unresolved_record_separated",
                    "an unresolved comparison must retain overlapping enclosures",
                )
        return self


__all__ = [
    "MAX_ENCLOSURE_COMPONENT_DIGITS",
    "MAX_RANGE_LENGTH",
    "MAX_SIMULTANEOUS_RADICANDS",
    "MAX_SURD_RADICAND",
    "MAX_SURD_SCALE_BITS",
    "NearestIntegerDistanceRequest",
    "NearestIntegerDistanceValue",
    "PositiveQuadraticIrrational",
    "RangeProfileRequest",
    "RangeProfileResult",
    "RangeProfileRow",
    "RecordMinimaRequest",
    "RecordMinimaResult",
    "RecordMinimumValue",
    "ScaledFloorRequest",
    "ScaledFloorResult",
    "ScaledFloorValue",
    "SimultaneousProductRequest",
    "SimultaneousProductResult",
    "bound_enclosure",
    "is_surd_radicand",
]
