"""Contracts for divisor-sum-product fibers and p-adic interval profiles."""

from __future__ import annotations

from typing import Self

from pydantic import Field, StrictInt, model_validator

from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer
from jacobian.math.number_theory._models import BoundedInteger, _validation_error

MAX_PREIMAGE_TARGET = 10_000_000
MAX_PREIMAGE_SOURCE = 3_162  # floor(sqrt(MAX_PREIMAGE_TARGET))
# A 252-digit endpoint leaves room for the exact total valuation (at most the
# interval length times the 2-adic logarithm of the endpoint) in the shared
# 256-digit canonical integer envelope.
MAX_INTERVAL_ENDPOINT_DIGITS = 252
MAX_INTERVAL_PRIME = 1_000_000
MAX_INTERVAL_PROFILE_ROWS = 1_024


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
    """A prime and the interval ``{start + 1, ..., start + length}``."""

    start: BoundedInteger = Field(
        description="Nonnegative canonical interval start m.",
        examples=["0"],
    )
    length: BoundedInteger = Field(
        description="Positive canonical interval length k.",
        examples=["10"],
    )
    prime: BoundedInteger = Field(
        description=f"Prime p between 2 and {MAX_INTERVAL_PRIME}.",
        examples=["2"],
    )

    @model_validator(mode="after")
    def require_admitted_interval(self) -> Self:
        start = parse_canonical_integer(self.start)
        length = parse_canonical_integer(self.length)
        prime = parse_canonical_integer(self.prime)
        if start < 0:
            raise _validation_error(
                "p_adic_interval_start_must_be_nonnegative",
                "start must be nonnegative",
            )
        if length < 1:
            raise _validation_error(
                "p_adic_interval_length_must_be_positive",
                "length must be positive",
            )
        if not 2 <= prime <= MAX_INTERVAL_PRIME:
            raise _validation_error(
                "p_adic_interval_prime_bound",
                f"prime must be between 2 and {MAX_INTERVAL_PRIME}",
            )
        from sympy import isprime

        if not isprime(prime):
            raise _validation_error(
                "p_adic_interval_prime_must_be_prime",
                "prime must be prime",
            )
        endpoint = start + length
        if len(str(endpoint)) > MAX_INTERVAL_ENDPOINT_DIGITS:
            raise _validation_error(
                "p_adic_interval_endpoint_digits",
                f"interval endpoint must have at most {MAX_INTERVAL_ENDPOINT_DIGITS} digits",
            )
        power_count = 0
        power = 1
        while power <= endpoint:
            power_count += 1
            power *= prime
        if power_count > MAX_INTERVAL_PROFILE_ROWS:
            raise _validation_error(
                "p_adic_interval_profile_row_bound",
                f"profile needs at most {MAX_INTERVAL_PROFILE_ROWS} rows",
            )
        return self


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
    "MAX_INTERVAL_ENDPOINT_DIGITS",
    "MAX_INTERVAL_PRIME",
    "MAX_INTERVAL_PROFILE_ROWS",
    "MAX_PREIMAGE_SOURCE",
    "MAX_PREIMAGE_TARGET",
    "DivisorSumProductPreimageRequest",
    "DivisorSumProductPreimageResult",
    "PAdicIntervalProfileRequest",
    "PAdicIntervalProfileResult",
    "PAdicIntervalProfileRow",
]
