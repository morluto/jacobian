"""Typed contracts and admission for bounded exact recurrence operations."""

from __future__ import annotations

import builtins
from itertools import pairwise
from typing import Literal, Self

from pydantic import ConfigDict, Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import (
    CanonicalInteger,
    CanonicalRational,
    require_bounded_rational,
)
from jacobian._models import StrictModel
from jacobian.math.polynomials.series._models import TruncatedSeries

MAX_LINEAR_RECURRENCE_ORDER = 16
MAX_LINEAR_RECURRENCE_INDEX = 512
MAX_LINEAR_RECURRENCE_REQUESTED_INDICES = 256
MAX_P_RECURSIVE_POLYNOMIAL_DEGREE = 16
MAX_RATIONAL_GENERATING_FUNCTION_DEGREE = 32
# The rational-series recurrence charges one coefficient construction plus
# every nonconstant denominator product. Its 250,000-unit work ledger therefore
# permits this many terms only for a constant denominator; higher-degree
# denominators admit proportionally shorter prefixes.
MAX_RATIONAL_SERIES_TRUNCATION_ORDER = 250_000
MAX_RATIONAL_SERIES_WORK_UNITS = 250_000
MAX_COMBINATORICS_INPUT_RATIONAL_DIGITS = 64
MAX_COMBINATORICS_RESULT_RATIONAL_DIGITS = 32_768
MAX_FIBONACCI_INDEX = 10_000


def _recurrence_validation_error(message: str) -> PydanticCustomError:
    """Return recurrence-family stable errors without shared heuristic parsing."""
    lowered = message.lower()
    code = "combinatorics.invariant"
    for marker, candidate in (
        ("recurrence", "combinatorics.recurrence_invariant"),
        ("residual", "combinatorics.recurrence_invariant"),
        ("polynomial", "combinatorics.polynomial_invariant"),
        ("requested index", "combinatorics.result_bound"),
        ("rational", "combinatorics.rational_bound"),
        ("result", "combinatorics.result_bound"),
    ):
        if marker in lowered:
            code = candidate
            break
    return PydanticCustomError(code, message, {})


def _require_bounded_rational(
    value: CanonicalRational,
    *,
    max_digits: int,
    label: str,
) -> None:
    """Adapt shared exact-value bounds into recurrence contract errors."""
    try:
        require_bounded_rational(value, max_digits=max_digits, label=label)
    except builtins.ValueError as exc:
        raise _recurrence_validation_error(
            f"{label} exceeds the {max_digits}-digit bound"
        ) from exc


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
        raise _recurrence_validation_error(
            f"{label} must omit trailing zero coefficients"
        )


class FibonacciPairResult(StrictModel):
    """Two consecutive Fibonacci values forming one recurrence boundary."""

    n: StrictInt = Field(ge=0, le=MAX_FIBONACCI_INDEX)
    f_n: CanonicalInteger
    f_n_plus_one: CanonicalInteger


class FibonacciPairRequest(StrictModel):
    n: StrictInt = Field(ge=0, le=MAX_FIBONACCI_INDEX)


class LinearRecurrenceEvaluationRequest(StrictModel):
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
            raise _recurrence_validation_error(
                "initial_values length must equal the recurrence order"
            )
        if self.scope == "PREFIX":
            if self.term_count is None or self.indices:
                raise _recurrence_validation_error(
                    "PREFIX scope requires term_count and forbids indices"
                )
        else:
            if self.term_count is not None or not self.indices:
                raise _recurrence_validation_error(
                    "INDICES scope requires indices and forbids term_count"
                )
            if any(
                index < 0 or index > MAX_LINEAR_RECURRENCE_INDEX
                for index in self.indices
            ):
                raise _recurrence_validation_error(
                    f"indices must lie between 0 and {MAX_LINEAR_RECURRENCE_INDEX}"
                )
            if any(left >= right for left, right in pairwise(self.indices)):
                raise _recurrence_validation_error(
                    "indices must be strictly increasing"
                )
        return self


class IndexedRationalValue(StrictModel):
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


class LinearRecurrenceEvaluationResult(StrictModel):
    coefficient_convention: Literal["A_N_EQUALS_SUM_C_J_TIMES_A_N_MINUS_J_FOR_J_FROM_1"]
    scope: Literal["PREFIX", "INDICES"]
    values: tuple[IndexedRationalValue, ...] = Field(
        min_length=1,
        max_length=MAX_LINEAR_RECURRENCE_INDEX + 1,
    )

    @classmethod
    def _from_kernel(
        cls,
        *,
        coefficient_convention: Literal[
            "A_N_EQUALS_SUM_C_J_TIMES_A_N_MINUS_J_FOR_J_FROM_1"
        ],
        scope: Literal["PREFIX", "INDICES"],
        values: tuple[IndexedRationalValue, ...],
    ) -> Self:
        return cls.model_construct(
            coefficient_convention=coefficient_convention,
            scope=scope,
            values=values,
        )

    @model_validator(mode="after")
    def require_canonical_projection(self) -> Self:
        indices = tuple(item.index for item in self.values)
        if any(left >= right for left, right in pairwise(indices)):
            raise _recurrence_validation_error(
                "result indices must be strictly increasing"
            )
        if self.scope == "PREFIX" and indices != tuple(range(len(indices))):
            raise _recurrence_validation_error(
                "PREFIX results must contain consecutive indices from zero"
            )
        return self


class PolynomialCoefficientRecurrenceEvaluationRequest(StrictModel):
    """Evaluate a bounded exact polynomial-coefficient linear recurrence.

    ``coefficient_polynomials[j]`` is the ascending coefficient vector of
    ``p_j(n)`` in ``sum_{j=0}^d p_j(n) a[n-j] = 0``.  The initial vector is
    exactly ``a[0], ..., a[d-1]``.
    """

    coefficient_polynomials: tuple[tuple[CanonicalRational, ...], ...] = Field(
        min_length=2, max_length=MAX_LINEAR_RECURRENCE_ORDER + 1
    )
    initial_values: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_LINEAR_RECURRENCE_ORDER
    )
    coefficient_convention: Literal[
        "SUM_P_J_OF_N_TIMES_A_N_MINUS_J_EQUALS_ZERO_FOR_J_FROM_0"
    ]
    polynomial_convention: Literal["ASCENDING_POWERS_OF_N"]
    scope: Literal["PREFIX", "INDICES"]
    term_count: StrictInt | None = Field(
        default=None, ge=1, le=MAX_LINEAR_RECURRENCE_INDEX + 1
    )
    indices: tuple[StrictInt, ...] = Field(
        default=(), max_length=MAX_LINEAR_RECURRENCE_REQUESTED_INDICES
    )

    @model_validator(mode="after")
    def require_bounded_regular_scope(self) -> Self:
        order = len(self.coefficient_polynomials) - 1
        if len(self.initial_values) != order:
            raise _recurrence_validation_error(
                "initial_values length must equal the recurrence order"
            )
        if self.scope == "PREFIX":
            if self.term_count is None or self.indices:
                raise _recurrence_validation_error(
                    "PREFIX scope requires term_count and forbids indices"
                )
        else:
            if self.term_count is not None or not self.indices:
                raise _recurrence_validation_error(
                    "INDICES scope requires indices and forbids term_count"
                )
            if any(
                index < 0 or index > MAX_LINEAR_RECURRENCE_INDEX
                for index in self.indices
            ):
                raise _recurrence_validation_error(
                    "indices are outside the recurrence bound"
                )
            if any(left >= right for left, right in pairwise(self.indices)):
                raise _recurrence_validation_error(
                    "indices must be strictly increasing"
                )
        return self


class PolynomialCoefficientRecurrenceEvaluationResult(StrictModel):
    coefficient_convention: Literal[
        "SUM_P_J_OF_N_TIMES_A_N_MINUS_J_EQUALS_ZERO_FOR_J_FROM_0"
    ]
    polynomial_convention: Literal["ASCENDING_POWERS_OF_N"]
    scope: Literal["PREFIX", "INDICES"]
    recurrence_order: StrictInt = Field(ge=1, le=MAX_LINEAR_RECURRENCE_ORDER)
    values: tuple[IndexedRationalValue, ...] = Field(
        min_length=1, max_length=MAX_LINEAR_RECURRENCE_INDEX + 1
    )

    @classmethod
    def _from_kernel(
        cls,
        *,
        coefficient_convention: Literal[
            "SUM_P_J_OF_N_TIMES_A_N_MINUS_J_EQUALS_ZERO_FOR_J_FROM_0"
        ],
        polynomial_convention: Literal["ASCENDING_POWERS_OF_N"],
        scope: Literal["PREFIX", "INDICES"],
        recurrence_order: int,
        values: tuple[IndexedRationalValue, ...],
    ) -> Self:
        return cls.model_construct(
            coefficient_convention=coefficient_convention,
            polynomial_convention=polynomial_convention,
            scope=scope,
            recurrence_order=recurrence_order,
            values=values,
        )

    @model_validator(mode="after")
    def require_canonical_projection(self) -> Self:
        indices = tuple(item.index for item in self.values)
        if any(left >= right for left, right in pairwise(indices)):
            raise _recurrence_validation_error(
                "result indices must be strictly increasing"
            )
        if self.scope == "PREFIX" and indices != tuple(range(len(indices))):
            raise _recurrence_validation_error(
                "PREFIX results must contain consecutive indices from zero"
            )
        return self


class RationalGeneratingFunctionCoefficientsRequest(StrictModel):
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


class RationalGeneratingFunctionCoefficientsResult(StrictModel):
    """A rational-function presentation bound to its canonical series value.

    The source presentation is retained so a consumer can check the defining
    congruence.  Parsing this value only checks its shape and scalar bounds;
    the mathematical relation belongs to
    :func:`verify_rational_generating_function_coefficients`.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "numerator": [{"num": "1", "den": "1"}],
                    "denominator": [
                        {"num": "1", "den": "1"},
                        {"num": "-1", "den": "1"},
                    ],
                    "coefficient_convention": "ASCENDING_POWERS_OF_X",
                    "expansion_point": "0",
                    "truncation_order": 3,
                    "series": {
                        "variable": "x",
                        "truncation_order": 3,
                        "coefficients": [
                            {"num": "1", "den": "1"},
                            {"num": "1", "den": "1"},
                            {"num": "1", "den": "1"},
                        ],
                    },
                }
            ]
        }
    )

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
    series: TruncatedSeries

    @classmethod
    def _from_kernel(
        cls,
        *,
        numerator: tuple[CanonicalRational, ...],
        denominator: tuple[CanonicalRational, ...],
        coefficient_convention: Literal["ASCENDING_POWERS_OF_X"],
        expansion_point: Literal["0"],
        truncation_order: int,
        coefficients: tuple[CanonicalRational, ...],
    ) -> Self:
        return cls.model_construct(
            numerator=numerator,
            denominator=denominator,
            coefficient_convention=coefficient_convention,
            expansion_point=expansion_point,
            truncation_order=truncation_order,
            series=TruncatedSeries(
                variable="x",
                truncation_order=truncation_order,
                coefficients=coefficients,
            ),
        )

    @model_validator(mode="after")
    def require_exact_finite_truncation(self) -> Self:
        _require_canonical_polynomial(self.numerator, label="numerator coefficient")
        _require_canonical_polynomial(self.denominator, label="denominator coefficient")
        if self.series.variable != "x":
            raise _recurrence_validation_error(
                "generating-function series variable must be x"
            )
        if self.series.truncation_order != self.truncation_order:
            raise _recurrence_validation_error(
                "series truncation order must equal truncation_order"
            )
        if len(self.series.coefficients) != self.truncation_order:
            raise _recurrence_validation_error(
                "series coefficients must equal truncation_order"
            )
        for value in self.series.coefficients:
            _require_bounded_rational(
                value,
                max_digits=MAX_COMBINATORICS_RESULT_RATIONAL_DIGITS,
                label="series coefficient",
            )
        return self


__all__ = [
    "FibonacciPairRequest",
    "FibonacciPairResult",
    "IndexedRationalValue",
    "LinearRecurrenceEvaluationRequest",
    "LinearRecurrenceEvaluationResult",
    "PolynomialCoefficientRecurrenceEvaluationRequest",
    "PolynomialCoefficientRecurrenceEvaluationResult",
    "RationalGeneratingFunctionCoefficientsRequest",
    "RationalGeneratingFunctionCoefficientsResult",
]
