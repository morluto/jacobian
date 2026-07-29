"""Named Pydantic wire contracts for exact combinatorics capabilities."""

from __future__ import annotations

from fractions import Fraction
from itertools import pairwise
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian.contracts.exact import CanonicalInteger, CanonicalRational
from jacobian.contracts.results import ContractModel

_MAX_N = 1_000
_MAX_PARTS = 256
_MAX_PARTITION_N = 30
_MAX_ENUMERATED_PARTITIONS = 10_000
MAX_LINEAR_RECURRENCE_ORDER = 16
MAX_LINEAR_RECURRENCE_INDEX = 512
MAX_LINEAR_RECURRENCE_REQUESTED_INDICES = 256
MAX_RATIONAL_GENERATING_FUNCTION_DEGREE = 32
MAX_RATIONAL_SERIES_TRUNCATION_ORDER = 512
MAX_COMBINATORICS_INPUT_RATIONAL_DIGITS = 64
MAX_COMBINATORICS_RESULT_RATIONAL_DIGITS = 32_768


def _require_bounded_rational(
    value: CanonicalRational,
    *,
    max_digits: int,
    label: str,
) -> Fraction:
    fraction = value.as_fraction()
    if (
        len(str(abs(fraction.numerator))) > max_digits
        or len(str(fraction.denominator)) > max_digits
    ):
        raise ValueError(f"{label} exceeds the {max_digits}-digit bound")
    return fraction


def _require_canonical_polynomial(
    coefficients: tuple[CanonicalRational, ...],
    *,
    label: str,
) -> None:
    for coefficient in coefficients:
        _require_bounded_rational(
            coefficient,
            max_digits=MAX_COMBINATORICS_INPUT_RATIONAL_DIGITS,
            label=label,
        )
    if len(coefficients) > 1 and coefficients[-1].as_fraction() == 0:
        raise ValueError(f"{label} must omit trailing zero coefficients")


class NonnegativeIntegerRequest(ContractModel):
    n: StrictInt = Field(ge=0, le=_MAX_N)


class NonnegativePairRequest(ContractModel):
    n: StrictInt = Field(ge=0, le=_MAX_N)
    k: StrictInt = Field(ge=0, le=_MAX_N)


class IntegerListRequest(ContractModel):
    values: tuple[CanonicalInteger, ...] = Field(min_length=1, max_length=_MAX_PARTS)

    @model_validator(mode="after")
    def require_nonnegative_parts(self) -> Self:
        if any(int(v) < 0 for v in self.values):
            raise ValueError("integer list values must be nonnegative")
        return self


class IntegerResult(ContractModel):
    value: CanonicalInteger


class RationalResult(ContractModel):
    value: CanonicalRational


class FibonacciPairResult(ContractModel):
    """Two consecutive Fibonacci values forming one recurrence boundary."""

    n: StrictInt = Field(ge=0, le=10_000)
    f_n: CanonicalInteger
    f_n_plus_one: CanonicalInteger


class FibonacciPairRequest(ContractModel):
    n: StrictInt = Field(ge=0, le=10_000)


class IntegerPartitionEnumerationRequest(ContractModel):
    """Enumerate every partition of n containing at most max_parts summands."""

    n: StrictInt = Field(ge=0, le=_MAX_PARTITION_N)
    max_parts: StrictInt = Field(ge=1, le=_MAX_PARTITION_N)


class IntegerPartitionEnumerationResult(ContractModel):
    """Complete canonical partition enumeration for one bounded request."""

    n: StrictInt = Field(ge=0, le=_MAX_PARTITION_N)
    max_parts: StrictInt = Field(ge=1, le=_MAX_PARTITION_N)
    partitions: tuple[tuple[StrictInt, ...], ...] = Field(
        max_length=_MAX_ENUMERATED_PARTITIONS
    )

    @model_validator(mode="after")
    def require_canonical_complete_items(self) -> Self:
        previous: tuple[int, ...] | None = None
        for partition in self.partitions:
            if len(partition) > self.max_parts:
                raise ValueError("partition exceeds max_parts")
            if any(part <= 0 for part in partition):
                raise ValueError("partition parts must be positive")
            if tuple(sorted(partition, reverse=True)) != partition:
                raise ValueError("partition parts must be nonincreasing")
            if sum(partition) != self.n:
                raise ValueError("partition parts must sum to n")
            if previous is not None and previous <= partition:
                raise ValueError(
                    "partitions must be unique in descending lexicographic order"
                )
            previous = tuple(partition)
        if self.n == 0 and self.partitions != ((),):
            raise ValueError("zero has exactly one empty partition")
        return self


class LinearRecurrenceEvaluationRequest(ContractModel):
    """Evaluate a bounded exact constant-coefficient recurrence.

    ``coefficients[j - 1]`` multiplies ``a[n - j]``. The initial vector is
    exactly ``a[0], ..., a[d - 1]`` for recurrence order ``d``.
    """

    coefficients: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_LINEAR_RECURRENCE_ORDER,
    )
    initial_values: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_LINEAR_RECURRENCE_ORDER,
    )
    coefficient_convention: Literal["A_N_EQUALS_SUM_C_J_TIMES_A_N_MINUS_J_FOR_J_FROM_1"]
    scope: Literal["PREFIX", "INDICES"]
    term_count: StrictInt | None = Field(
        default=None,
        ge=1,
        le=MAX_LINEAR_RECURRENCE_INDEX + 1,
    )
    indices: tuple[StrictInt, ...] = Field(
        default=(),
        max_length=MAX_LINEAR_RECURRENCE_REQUESTED_INDICES,
    )

    @model_validator(mode="after")
    def require_bounded_explicit_scope(self) -> Self:
        if len(self.initial_values) != len(self.coefficients):
            raise ValueError("initial_values length must equal the recurrence order")
        for label, values in (
            ("recurrence coefficient", self.coefficients),
            ("recurrence initial value", self.initial_values),
        ):
            for value in values:
                _require_bounded_rational(
                    value,
                    max_digits=MAX_COMBINATORICS_INPUT_RATIONAL_DIGITS,
                    label=label,
                )
        if self.scope == "PREFIX":
            if self.term_count is None or self.indices:
                raise ValueError("PREFIX scope requires term_count and forbids indices")
        else:
            if self.term_count is not None or not self.indices:
                raise ValueError(
                    "INDICES scope requires indices and forbids term_count"
                )
            if any(
                index < 0 or index > MAX_LINEAR_RECURRENCE_INDEX
                for index in self.indices
            ):
                raise ValueError(
                    f"indices must lie between 0 and {MAX_LINEAR_RECURRENCE_INDEX}"
                )
            if any(left >= right for left, right in pairwise(self.indices)):
                raise ValueError("indices must be strictly increasing")
        return self


class IndexedRationalValue(ContractModel):
    index: StrictInt = Field(ge=0, le=MAX_LINEAR_RECURRENCE_INDEX)
    value: CanonicalRational

    @model_validator(mode="after")
    def require_bounded_value(self) -> Self:
        _require_bounded_rational(
            self.value,
            max_digits=MAX_COMBINATORICS_RESULT_RATIONAL_DIGITS,
            label="recurrence result",
        )
        return self


class LinearRecurrenceEvaluationResult(ContractModel):
    coefficient_convention: Literal["A_N_EQUALS_SUM_C_J_TIMES_A_N_MINUS_J_FOR_J_FROM_1"]
    scope: Literal["PREFIX", "INDICES"]
    values: tuple[IndexedRationalValue, ...] = Field(
        min_length=1,
        max_length=MAX_LINEAR_RECURRENCE_INDEX + 1,
    )
    replay_prefix: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_LINEAR_RECURRENCE_INDEX + 1,
    )
    replay_scope_end: StrictInt = Field(ge=0, le=MAX_LINEAR_RECURRENCE_INDEX)
    exactness: Literal["EXACT_RATIONAL"] = "EXACT_RATIONAL"
    determinism: Literal["DETERMINISTIC"] = "DETERMINISTIC"
    backend: Literal["sympy"] = "sympy"
    backend_version: Literal["1.14.0"] = "1.14.0"
    verification: Literal["UNVERIFIED"] = "UNVERIFIED"

    @model_validator(mode="after")
    def require_complete_replay_prefix(self) -> Self:
        if len(self.replay_prefix) != self.replay_scope_end + 1:
            raise ValueError(
                "replay_prefix must cover indices 0 through replay_scope_end"
            )
        for value in self.replay_prefix:
            _require_bounded_rational(
                value,
                max_digits=MAX_COMBINATORICS_RESULT_RATIONAL_DIGITS,
                label="recurrence replay value",
            )
        indices = tuple(item.index for item in self.values)
        if any(left >= right for left, right in pairwise(indices)):
            raise ValueError("result indices must be strictly increasing")
        if indices[-1] != self.replay_scope_end:
            raise ValueError("the greatest requested index must bind replay_scope_end")
        if any(item.value != self.replay_prefix[item.index] for item in self.values):
            raise ValueError("indexed values must match the recurrence replay prefix")
        if self.scope == "PREFIX" and indices != tuple(range(len(indices))):
            raise ValueError(
                "PREFIX results must contain consecutive indices from zero"
            )
        return self


class RationalGeneratingFunctionCoefficientsRequest(ContractModel):
    """Expand N(x)/D(x) at zero through one explicit finite order."""

    numerator: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_RATIONAL_GENERATING_FUNCTION_DEGREE + 1,
    )
    denominator: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_RATIONAL_GENERATING_FUNCTION_DEGREE + 1,
    )
    coefficient_convention: Literal["ASCENDING_POWERS_OF_X"]
    expansion_point: Literal["0"]
    truncation_order: StrictInt = Field(
        ge=1,
        le=MAX_RATIONAL_SERIES_TRUNCATION_ORDER,
    )

    @model_validator(mode="after")
    def require_regular_canonical_input(self) -> Self:
        _require_canonical_polynomial(self.numerator, label="numerator coefficient")
        _require_canonical_polynomial(
            self.denominator,
            label="denominator coefficient",
        )
        if self.denominator[0].as_fraction() == 0:
            raise ValueError("denominator constant coefficient must be nonzero")
        return self


class RationalGeneratingFunctionCoefficientsResult(ContractModel):
    coefficient_convention: Literal["ASCENDING_POWERS_OF_X"]
    expansion_point: Literal["0"]
    truncation_order: StrictInt = Field(
        ge=1,
        le=MAX_RATIONAL_SERIES_TRUNCATION_ORDER,
    )
    coefficients: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_RATIONAL_SERIES_TRUNCATION_ORDER,
    )
    residual_congruence: Literal[
        "DENOMINATOR_TIMES_SERIES_MINUS_NUMERATOR_IS_ZERO_MOD_X_TO_ORDER"
    ]
    residual_coefficients: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_RATIONAL_SERIES_TRUNCATION_ORDER,
    )
    exactness: Literal["EXACT_RATIONAL"] = "EXACT_RATIONAL"
    determinism: Literal["DETERMINISTIC"] = "DETERMINISTIC"
    backend: Literal["sympy"] = "sympy"
    backend_version: Literal["1.14.0"] = "1.14.0"
    verification: Literal["UNVERIFIED"] = "UNVERIFIED"

    @model_validator(mode="after")
    def require_exact_finite_truncation(self) -> Self:
        if (
            len(self.coefficients) != self.truncation_order
            or len(self.residual_coefficients) != self.truncation_order
        ):
            raise ValueError(
                "coefficient and residual vectors must equal truncation_order"
            )
        for label, values in (
            ("series coefficient", self.coefficients),
            ("series residual", self.residual_coefficients),
        ):
            for value in values:
                _require_bounded_rational(
                    value,
                    max_digits=MAX_COMBINATORICS_RESULT_RATIONAL_DIGITS,
                    label=label,
                )
        if any(value.as_fraction() != 0 for value in self.residual_coefficients):
            raise ValueError("residual coefficients must vanish through the truncation")
        return self
