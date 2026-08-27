"""Contracts for divisor-sum-product fibers and p-adic interval profiles."""

from __future__ import annotations

from typing import Self

from pydantic import ConfigDict, Field, StrictInt, model_validator

from jacobian._models import StrictModel
from jacobian.canonical import CanonicalLimits, parse_canonical_integer
from jacobian.math.number_theory._models import (
    MAX_INTEGER_DIGITS,
    BoundedInteger,
    _validation_error,
)

MAX_PREIMAGE_TARGET = 10_000_000
MAX_PREIMAGE_SOURCE = 3_162  # floor(sqrt(MAX_PREIMAGE_TARGET))
MAX_INTERVAL_PROFILE_ROWS = 1_024
MAX_INTERVAL_PROFILE_WORK = 3 * MAX_INTERVAL_PROFILE_ROWS
MAX_INTERVAL_PROFILE_RESULT_BYTES = CanonicalLimits().max_output_bytes
PRIMALITY_WORK_DIGIT_EXPONENT: int = 3

_P_ADIC_REQUEST_DESCRIPTION = (
    "A prime p and the interval {start + 1, ..., start + length}. Let U = "
    "start + length. Admission evaluates the exact valuation profile for the "
    f"coupled endpoint U, with at most {MAX_INTERVAL_PROFILE_ROWS} visited powers "
    f"and combined bounded arithmetic work satisfying 3 * power_count + "
    f"decimal_digits(prime)^{PRIMALITY_WORK_DIGIT_EXPONENT} <= "
    f"{MAX_INTERVAL_PROFILE_WORK}, and admits "
    f"only when the exact total valuation fits {MAX_INTEGER_DIGITS} canonical "
    "digits and the complete canonical result fits the output envelope. There "
    "is no standalone endpoint digit ceiling: useful endpoints are admitted "
    "when their exact sum, work, and result fit these bounds."
)


class DivisorSumProductPreimageRequest(StrictModel):
    """One positive target for the map ``n -> n * sigma(n)``."""

    target: BoundedInteger = Field(
        description=(
            "Positive canonical target m. The complete source fiber is searched "
            f"through floor(sqrt(m)) for m <= {MAX_PREIMAGE_TARGET}."
        ),
        examples=["336"],
    )

    @model_validator(mode="after")
    def require_admitted_target(self) -> Self:
        target = parse_canonical_integer(self.target)
        if not 1 <= target <= MAX_PREIMAGE_TARGET:
            raise _validation_error(
                "divisor_sum_product_target_bound",
                f"target must be between 1 and {MAX_PREIMAGE_TARGET}",
            )
        return self


class DivisorSumProductPreimageResult(StrictModel):
    """The complete increasing fiber of ``n -> n * sigma(n)``."""

    target: BoundedInteger
    preimages: tuple[BoundedInteger, ...] = Field(max_length=MAX_PREIMAGE_SOURCE)
    count: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def require_canonical_fiber_shape(self) -> Self:
        target = parse_canonical_integer(self.target)
        values = tuple(parse_canonical_integer(value) for value in self.preimages)
        if not 1 <= target <= MAX_PREIMAGE_TARGET:
            raise _validation_error(
                "divisor_sum_product_target_bound",
                f"target must be between 1 and {MAX_PREIMAGE_TARGET}",
            )
        if any(value < 1 for value in values):
            raise _validation_error(
                "divisor_sum_product_preimages_must_be_positive",
                "preimages must be positive",
            )
        if values != tuple(sorted(set(values))):
            raise _validation_error(
                "divisor_sum_product_preimages_must_be_sorted",
                "preimages must be strictly increasing",
            )
        if self.count != len(values):
            raise _validation_error(
                "divisor_sum_product_count_mismatch",
                "count must equal the number of preimages",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: DivisorSumProductPreimageRequest,
        *,
        preimages: tuple[int, ...],
    ) -> Self:
        return cls(
            target=request.target,
            preimages=tuple(str(value) for value in preimages),
            count=len(preimages),
        )


class PAdicIntervalProfileRequest(StrictModel):
    """A prime and one exactly admitted interval valuation profile."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": _P_ADIC_REQUEST_DESCRIPTION,
            "examples": [{"start": "0", "length": "10", "prime": "2"}],
            "endpoint_sum_admission": {
                "endpoint": "start + length",
                "max_profile_powers": MAX_INTERVAL_PROFILE_ROWS,
                "max_profile_work_units": MAX_INTERVAL_PROFILE_WORK,
                "primality_work_units": f"decimal_digits(prime)^{PRIMALITY_WORK_DIGIT_EXPONENT}",
                "total_valuation_max_digits": MAX_INTEGER_DIGITS,
                "canonical_result_max_bytes": MAX_INTERVAL_PROFILE_RESULT_BYTES,
            },
        }
    )

    start: BoundedInteger = Field(
        description=(
            "Nonnegative canonical interval start m. The coupled endpoint "
            "start + length is admitted from the exact profile bounds."
        ),
        examples=["0"],
    )
    length: BoundedInteger = Field(
        description=(
            "Positive canonical interval length k. Together with start, its "
            "endpoint U = start + length must fit the exact valuation-sum, "
            "profile-work, and canonical-result bounds described above."
        ),
        examples=["10"],
    )
    prime: BoundedInteger = Field(
        description="Prime p represented by a canonical decimal integer.",
        examples=["2"],
    )


class PAdicIntervalProfileRow(StrictModel):
    """One nonempty valuation class in an interval histogram."""

    valuation: StrictInt = Field(ge=0, le=MAX_INTERVAL_PROFILE_ROWS)
    count: BoundedInteger

    @model_validator(mode="after")
    def require_positive_count(self) -> Self:
        if parse_canonical_integer(self.count) < 1:
            raise _validation_error(
                "p_adic_interval_profile_count_must_be_positive",
                "profile row counts must be positive",
            )
        return self


class PAdicIntervalProfileResult(StrictModel):
    """The exact valuation histogram for one consecutive integer interval."""

    start: BoundedInteger
    length: BoundedInteger
    prime: BoundedInteger
    rows: tuple[PAdicIntervalProfileRow, ...] = Field(
        max_length=MAX_INTERVAL_PROFILE_ROWS
    )
    total_valuation: BoundedInteger
    maximum_valuation: StrictInt = Field(ge=0, le=MAX_INTERVAL_PROFILE_ROWS)

    @model_validator(mode="after")
    def require_profile_shape(self) -> Self:
        start = parse_canonical_integer(self.start)
        length = parse_canonical_integer(self.length)
        if start < 0 or length < 1:
            raise _validation_error(
                "p_adic_interval_profile_source_shape",
                "profile source must have nonnegative start and positive length",
            )
        valuations = tuple(row.valuation for row in self.rows)
        if valuations != tuple(sorted(set(valuations))):
            raise _validation_error(
                "p_adic_interval_profile_rows_must_be_sorted",
                "profile rows must have strictly increasing valuations",
            )
        if valuations and self.maximum_valuation != valuations[-1]:
            raise _validation_error(
                "p_adic_interval_profile_maximum_mismatch",
                "maximum_valuation must equal the last profile valuation",
            )
        if not valuations and self.maximum_valuation != 0:
            raise _validation_error(
                "p_adic_interval_profile_maximum_mismatch",
                "an empty profile must have maximum_valuation zero",
            )
        counts = tuple(parse_canonical_integer(row.count) for row in self.rows)
        if sum(counts) != length:
            raise _validation_error(
                "p_adic_interval_profile_count_sum_mismatch",
                "profile row counts must sum to length",
            )
        total_valuation = parse_canonical_integer(self.total_valuation)
        if total_valuation < 0:
            raise _validation_error(
                "p_adic_interval_profile_total_must_be_nonnegative",
                "total valuation must be nonnegative",
            )
        if total_valuation != sum(
            row.valuation * count for row, count in zip(self.rows, counts, strict=True)
        ):
            raise _validation_error(
                "p_adic_interval_profile_total_mismatch",
                "total_valuation must equal the weighted profile sum",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: PAdicIntervalProfileRequest,
        *,
        rows: tuple[PAdicIntervalProfileRow, ...],
        total_valuation: int,
        maximum_valuation: int,
    ) -> Self:
        return cls(
            start=request.start,
            length=request.length,
            prime=request.prime,
            rows=rows,
            total_valuation=str(total_valuation),
            maximum_valuation=maximum_valuation,
        )


__all__ = [
    "MAX_INTERVAL_PROFILE_RESULT_BYTES",
    "MAX_INTERVAL_PROFILE_ROWS",
    "MAX_INTERVAL_PROFILE_WORK",
    "MAX_PREIMAGE_SOURCE",
    "MAX_PREIMAGE_TARGET",
    "PRIMALITY_WORK_DIGIT_EXPONENT",
    "DivisorSumProductPreimageRequest",
    "DivisorSumProductPreimageResult",
    "PAdicIntervalProfileRequest",
    "PAdicIntervalProfileResult",
    "PAdicIntervalProfileRow",
]
