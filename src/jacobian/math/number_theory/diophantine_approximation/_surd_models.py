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
# Complete range operations expand a rectangular table.  Admit work and
# retained allocation before any row or exact interval is materialized.  One
# allocation unit conservatively reserves an exact scalar bit or a fixed
# structural slot; transports own their encoded-byte ceilings separately.
MAX_SURD_RANGE_WORK = 32_000_000
MAX_SURD_RANGE_INTERMEDIATE_BITS = 64_000_000
MAX_SURD_RANGE_ALLOCATION_UNITS = 16 * 1024 * 1024

_SURD_AXIS_DESCRIPTION = (
    "Ordered tuple of distinct nonsquare integer radicands in the admitted "
    f"range 2..{MAX_SURD_RADICAND}."
)
_SURD_SCALAR_RADICAND_DESCRIPTION = (
    "Nonsquare integer radicand in the admitted range "
    f"2..{MAX_SURD_RADICAND}."
)


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
    radicand: StrictInt = Field(
        ge=2,
        le=MAX_SURD_RADICAND,
        description=_SURD_SCALAR_RADICAND_DESCRIPTION,
    )


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
    def require_canonical_bracket_shape(self) -> Self:
        if not is_surd_radicand(self.radicand):
            raise _validation_error(
                "diophantine.surd_radicand_must_not_be_square",
                "a quadratic irrational requires a nonsquare radicand",
            )
        if self.floor < 0 or self.ceiling != self.floor + 1:
            raise _validation_error(
                "diophantine.scaled_floor_bracket_shape",
                "floor and ceiling must be consecutive nonnegative integers",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        multiplier: int,
        radicand: int,
        floor: int,
        ceiling: int,
        square_lower: int,
        square_upper: int,
    ) -> Self:
        """Construct an exact row after the owner kernel established it."""

        return cls.model_construct(
            multiplier=multiplier,
            radicand=radicand,
            floor=floor,
            ceiling=ceiling,
            square_lower=square_lower,
            square_upper=square_upper,
        )


class NearestIntegerDistanceRequest(StrictModel):
    """Certify ``||n sqrt(d)||`` at a requested binary precision."""

    multiplier: ExactInteger = Field(
        ge=1,
        description="Positive exact multiplier; computation admits at most 4096 bits.",
    )
    radicand: StrictInt = Field(
        ge=2,
        le=MAX_SURD_RADICAND,
        description=_SURD_SCALAR_RADICAND_DESCRIPTION,
    )
    scale_bits: StrictInt = Field(ge=1, le=MAX_SURD_SCALE_BITS)


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
    def require_canonical_distance_shape(self) -> Self:
        if not is_surd_radicand(self.radicand):
            raise _validation_error(
                "diophantine.surd_radicand_must_not_be_square",
                "a quadratic irrational requires a nonsquare radicand",
            )
        if self.floor < 0 or self.ceiling != self.floor + 1:
            raise _validation_error(
                "diophantine.nearest_integer_bracket_shape",
                "floor and ceiling must be consecutive nonnegative integers",
            )
        expected_nearest = self.floor if self.side == "FLOOR" else self.ceiling
        if self.nearest_integer != expected_nearest:
            raise _validation_error(
                "diophantine.nearest_integer_side_mismatch",
                "nearest_integer must agree with the certified branch",
            )
        if self.distance_enclosure.lower.num < 0:
            raise _validation_error(
                "diophantine.nearest_integer_negative_distance",
                "a nearest-integer distance enclosure must be nonnegative",
            )
        upper = self.distance_enclosure.upper
        if self.distance_upper_scaled < 1 or (
            upper.num * (2**self.scale_bits) != self.distance_upper_scaled * upper.den
        ):
            raise _validation_error(
                "diophantine.nearest_integer_upper_mismatch",
                "distance_upper_scaled must retain the enclosure upper endpoint",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        multiplier: int,
        radicand: int,
        floor: int,
        ceiling: int,
        nearest_integer: int,
        side: Literal["FLOOR", "CEILING"],
        scale_bits: int,
        distance_enclosure: ClosedRationalInterval,
        distance_upper_scaled: int,
    ) -> Self:
        """Construct a distance value after the exact kernel established it."""

        return cls.model_construct(
            multiplier=multiplier,
            radicand=radicand,
            floor=floor,
            ceiling=ceiling,
            nearest_integer=nearest_integer,
            side=side,
            scale_bits=scale_bits,
            distance_enclosure=distance_enclosure,
            distance_upper_scaled=distance_upper_scaled,
        )


class SimultaneousProductRequest(StrictModel):
    """Certify ``n * prod_i ||n sqrt(d_i)||`` over an ordered radicand axis."""

    multiplier: ExactInteger = Field(
        ge=1,
        description="Positive exact multiplier; computation admits at most 4096 bits.",
    )
    radicands: tuple[StrictInt, ...] = Field(
        min_length=1,
        max_length=MAX_SIMULTANEOUS_RADICANDS,
        description=_SURD_AXIS_DESCRIPTION,
    )
    scale_bits: StrictInt = Field(ge=1, le=MAX_SURD_SCALE_BITS)

    @model_validator(mode="after")
    def require_distinct_radicands(self) -> Self:
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
        if any(not is_surd_radicand(radicand) for radicand in self.radicands):
            raise _validation_error(
                "diophantine.surd_radicand_must_not_be_square",
                "a quadratic irrational requires nonsquare radicands",
            )
        if len(set(self.radicands)) != len(self.radicands):
            raise _validation_error(
                "diophantine.surd_radicands_must_be_distinct",
                "an ordered radicand axis requires distinct radicands",
            )
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
        if any(factor.scale_bits != self.scale_bits for factor in self.factors):
            raise _validation_error(
                "diophantine.product_factor_scale_mismatch",
                "every factor row must share the requested precision",
            )
        if self.product_enclosure.lower.num < 0:
            raise _validation_error(
                "diophantine.product_negative_enclosure",
                "a simultaneous product enclosure must be nonnegative",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        multiplier: int,
        radicands: tuple[int, ...],
        scale_bits: int,
        factors: tuple[NearestIntegerDistanceValue, ...],
        product_enclosure: ClosedRationalInterval,
    ) -> Self:
        """Construct a product result after the owner kernel established it."""

        return cls.model_construct(
            multiplier=multiplier,
            radicands=radicands,
            scale_bits=scale_bits,
            factors=factors,
            product_enclosure=product_enclosure,
        )


class RangeProfileRequest(StrictModel):
    """A complete certified row for every ``1 <= n <= limit``."""

    radicands: tuple[StrictInt, ...] = Field(
        min_length=1,
        max_length=MAX_SIMULTANEOUS_RADICANDS,
        description=_SURD_AXIS_DESCRIPTION,
    )
    limit: StrictInt = Field(ge=1, le=MAX_RANGE_LENGTH)
    scale_bits: StrictInt = Field(ge=1, le=MAX_SURD_SCALE_BITS)

    @model_validator(mode="after")
    def require_distinct_radicands(self) -> Self:
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

    @model_validator(mode="after")
    def require_factor_multiplier_alignment(self) -> Self:
        if any(factor.multiplier != self.multiplier for factor in self.factors):
            raise _validation_error(
                "diophantine.range_profile_factor_multiplier_mismatch",
                "every row factor must share its row multiplier",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        multiplier: int,
        factors: tuple[NearestIntegerDistanceValue, ...],
        product_enclosure: ClosedRationalInterval,
    ) -> Self:
        """Construct a row after the owner kernel established its bindings."""

        return cls.model_construct(
            multiplier=multiplier,
            factors=factors,
            product_enclosure=product_enclosure,
        )


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
        if any(not is_surd_radicand(radicand) for radicand in self.radicands):
            raise _validation_error(
                "diophantine.surd_radicand_must_not_be_square",
                "a quadratic irrational requires nonsquare radicands",
            )
        if len(set(self.radicands)) != len(self.radicands):
            raise _validation_error(
                "diophantine.surd_radicands_must_be_distinct",
                "an ordered radicand axis requires distinct radicands",
            )
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
            if any(factor.scale_bits != self.scale_bits for factor in row.factors):
                raise _validation_error(
                    "diophantine.range_profile_factor_scale_mismatch",
                    "every row factor must share the profile precision",
                )
            if row.product_enclosure.lower.num < 0:
                raise _validation_error(
                    "diophantine.range_profile_negative_product",
                    "a range product enclosure must be nonnegative",
                )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        radicands: tuple[int, ...],
        limit: int,
        scale_bits: int,
        rows: tuple[RangeProfileRow, ...],
    ) -> Self:
        """Construct a complete profile after the owner kernel established it."""

        return cls.model_construct(
            radicands=radicands,
            limit=limit,
            scale_bits=scale_bits,
            rows=rows,
        )


class RecordMinimaRequest(StrictModel):
    """Extract strict record minima from a certified simultaneous range."""

    radicands: tuple[StrictInt, ...] = Field(
        min_length=1,
        max_length=MAX_SIMULTANEOUS_RADICANDS,
        description=_SURD_AXIS_DESCRIPTION,
    )
    limit: StrictInt = Field(ge=1, le=MAX_RANGE_LENGTH)
    scale_bits: StrictInt = Field(ge=1, le=MAX_SURD_SCALE_BITS)

    @model_validator(mode="after")
    def require_distinct_radicands(self) -> Self:
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

    @classmethod
    def _from_kernel(
        cls,
        *,
        multiplier: int,
        product_enclosure: ClosedRationalInterval,
        incumbent_enclosure: ClosedRationalInterval,
    ) -> Self:
        """Construct one record after the owner kernel established it."""

        return cls.model_construct(
            multiplier=multiplier,
            product_enclosure=product_enclosure,
            incumbent_enclosure=incumbent_enclosure,
        )


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
    unresolved_multiplier: ExactInteger | None = None
    unresolved_product_enclosure: ClosedRationalInterval | None = None
    unresolved_incumbent_enclosure: ClosedRationalInterval | None = None

    @model_validator(mode="after")
    def require_surd_axis(self) -> Self:
        if any(not is_surd_radicand(radicand) for radicand in self.radicands):
            raise _validation_error(
                "diophantine.surd_radicand_must_not_be_square",
                "a quadratic irrational requires nonsquare radicands",
            )
        if len(set(self.radicands)) != len(self.radicands):
            raise _validation_error(
                "diophantine.surd_radicands_must_be_distinct",
                "an ordered radicand axis requires distinct radicands",
            )
        return self

    @model_validator(mode="after")
    def require_record_history(self) -> Self:
        if not self.records or self.records[0].multiplier != 1:
            raise _validation_error(
                "diophantine.record_sequence_must_start_at_one",
                "a finite record sequence must retain its first row at multiplier one",
            )
        if any(
            record.product_enclosure.lower.num < 0
            or record.incumbent_enclosure.lower.num < 0
            for record in self.records
        ) or (
            self.outcome == "UNRESOLVED"
            and self.unresolved_product_enclosure is not None
            and self.unresolved_product_enclosure.lower.num < 0
        ):
            raise _validation_error(
                "diophantine.record_negative_product_enclosure",
                "record product enclosures must be nonnegative",
            )
        if self.records[0].incumbent_enclosure != self.records[0].product_enclosure:
            raise _validation_error(
                "diophantine.first_record_incumbent_mismatch",
                "the first record's incumbent enclosure must equal its product",
            )
        if any(
            current.incumbent_enclosure != previous.product_enclosure
            for previous, current in zip(self.records, self.records[1:], strict=False)
        ):
            raise _validation_error(
                "diophantine.record_incumbent_source_mismatch",
                "each record must retain the prior record's product enclosure",
            )
        return self

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
            if self.unresolved_multiplier > self.limit:
                raise _validation_error(
                    "diophantine.unresolved_record_multiplier_out_of_range",
                    "the unresolved multiplier must lie in the declared range",
                )
            if (
                self.records
                and self.unresolved_multiplier <= self.records[-1].multiplier
            ):
                raise _validation_error(
                    "diophantine.unresolved_record_order",
                    "the unresolved multiplier must follow all certified records",
                )
        return self

    @model_validator(mode="after")
    def require_common_record_axis(self) -> Self:
        """Validate record provenance shared by complete and unresolved results."""

        if any(record.multiplier > self.limit for record in self.records):
            raise _validation_error(
                (
                    "diophantine.complete_record_multiplier_out_of_range"
                    if self.outcome == "COMPLETE"
                    else "diophantine.record_multiplier_out_of_range"
                ),
                "every record multiplier must lie in the declared range",
            )
        if any(
            current.multiplier <= previous.multiplier
            for previous, current in zip(self.records, self.records[1:], strict=False)
        ):
            raise _validation_error(
                (
                    "diophantine.complete_record_order"
                    if self.outcome == "COMPLETE"
                    else "diophantine.record_order"
                ),
                "record multipliers must be strictly increasing",
            )
        if (
            self.outcome == "COMPLETE"
            and self.finite_argmin is not None
            and self.finite_argmin > self.limit
        ):
            raise _validation_error(
                "diophantine.complete_record_argmin_out_of_range",
                "finite_argmin must lie in the declared range",
            )
        if self.outcome == "UNRESOLVED":
            assert self.unresolved_incumbent_enclosure is not None
            if (
                self.unresolved_incumbent_enclosure
                != self.records[-1].product_enclosure
            ):
                raise _validation_error(
                    "diophantine.unresolved_record_incumbent_mismatch",
                    "the unresolved incumbent must equal the last retained product",
                )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        radicands: tuple[int, ...],
        limit: int,
        scale_bits: int,
        outcome: Literal["COMPLETE", "UNRESOLVED"],
        records: tuple[RecordMinimumValue, ...],
        finite_argmin: int | None = None,
        unresolved_multiplier: int | None = None,
        unresolved_product_enclosure: ClosedRationalInterval | None = None,
        unresolved_incumbent_enclosure: ClosedRationalInterval | None = None,
    ) -> Self:
        """Construct a result after the owner kernel established its claims."""

        return cls.model_construct(
            radicands=radicands,
            limit=limit,
            scale_bits=scale_bits,
            outcome=outcome,
            records=records,
            finite_argmin=finite_argmin,
            unresolved_multiplier=unresolved_multiplier,
            unresolved_product_enclosure=unresolved_product_enclosure,
            unresolved_incumbent_enclosure=unresolved_incumbent_enclosure,
        )


__all__ = [
    "MAX_ENCLOSURE_COMPONENT_DIGITS",
    "MAX_RANGE_LENGTH",
    "MAX_SIMULTANEOUS_RADICANDS",
    "MAX_SURD_MULTIPLIER_BITS",
    "MAX_SURD_RADICAND",
    "MAX_SURD_RANGE_ALLOCATION_UNITS",
    "MAX_SURD_RANGE_INTERMEDIATE_BITS",
    "MAX_SURD_RANGE_WORK",
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
    "ScaledFloorValue",
    "SimultaneousProductRequest",
    "SimultaneousProductResult",
    "bound_enclosure",
    "is_surd_radicand",
]
