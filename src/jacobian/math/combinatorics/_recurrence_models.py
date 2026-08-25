"""Typed contracts and admission for bounded exact recurrence operations."""

from __future__ import annotations

import builtins
import math
from fractions import Fraction
from itertools import pairwise
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import (
    CanonicalInteger,
    CanonicalRational,
    require_bounded_rational,
)
from jacobian._models import StrictModel
from jacobian.canonical import (
    CanonicalLimits,
    canonicalize_json,
    format_canonical_integer,
)

MAX_LINEAR_RECURRENCE_ORDER = 16
MAX_LINEAR_RECURRENCE_INDEX = 512
MAX_LINEAR_RECURRENCE_REQUESTED_INDICES = 256
MAX_P_RECURSIVE_POLYNOMIAL_DEGREE = 16
MAX_RATIONAL_GENERATING_FUNCTION_DEGREE = 32
MAX_RATIONAL_SERIES_TRUNCATION_ORDER = 512
MAX_COMBINATORICS_INPUT_RATIONAL_DIGITS = 64
MAX_COMBINATORICS_RESULT_RATIONAL_DIGITS = 32_768
MAX_COMBINATORICS_RESULT_ARTIFACT_BYTES = 10 * 1024 * 1024
_LOG10_2 = math.log10(2)


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


def _fraction_wire(value: Fraction) -> dict[str, str]:
    return {
        "num": format_canonical_integer(value.numerator),
        "den": format_canonical_integer(value.denominator),
    }


def _require_bounded_fraction(
    value: Fraction,
    *,
    max_digits: int,
    label: str,
) -> None:
    if (
        len(format_canonical_integer(abs(value.numerator))) > max_digits
        or len(format_canonical_integer(value.denominator)) > max_digits
    ):
        raise _recurrence_validation_error(
            f"{label} exceeds the {max_digits}-digit bound"
        )


def _lower_decimal_digits(value: int) -> int:
    if value == 0:
        return 1
    return math.floor((abs(value).bit_length() - 1) * _LOG10_2) + 1


def _minimum_fraction_wire_bytes(value: Fraction) -> int:
    return (
        _lower_decimal_digits(value.numerator)
        + _lower_decimal_digits(value.denominator)
        + 20
    )


def _validate_result_inline_size(payload: dict[str, object]) -> None:
    try:
        canonicalize_json(
            payload,
            limits=CanonicalLimits(
                max_output_bytes=MAX_COMBINATORICS_RESULT_ARTIFACT_BYTES,
                max_integer_digits=MAX_COMBINATORICS_RESULT_RATIONAL_DIGITS,
            ),
        )
    except ValueError as exc:
        raise _recurrence_validation_error(
            "the exact combinatorics result exceeds the bounded result limit"
        ) from exc


def _recurrence_replay(
    coefficients: tuple[Fraction, ...],
    initial_values: tuple[Fraction, ...],
    end: int,
) -> list[Fraction]:
    replay = list(initial_values[: end + 1])
    while len(replay) <= end:
        replay.append(
            sum(
                (
                    coefficient * replay[len(replay) - offset]
                    for offset, coefficient in enumerate(coefficients, start=1)
                ),
                start=Fraction(),
            )
        )
    return replay


def _validate_recurrence_result_budget(
    *,
    coefficients: tuple[CanonicalRational, ...],
    initial_values: tuple[CanonicalRational, ...],
    coefficient_convention: str,
    scope: str,
    requested_indices: tuple[int, ...],
) -> None:
    replay = _recurrence_replay(
        tuple(value.as_fraction() for value in coefficients),
        tuple(value.as_fraction() for value in initial_values),
        requested_indices[-1],
    )
    minimum_size = sum(_minimum_fraction_wire_bytes(value) for value in replay)
    minimum_size += sum(
        _minimum_fraction_wire_bytes(replay[index]) for index in requested_indices
    )
    if minimum_size + 1_024 > MAX_COMBINATORICS_RESULT_ARTIFACT_BYTES:
        raise _recurrence_validation_error(
            "the exact combinatorics result exceeds the bounded result limit"
        )
    for value in replay:
        if any(
            _lower_decimal_digits(component) > MAX_COMBINATORICS_RESULT_RATIONAL_DIGITS
            for component in (value.numerator, value.denominator)
        ):
            raise _recurrence_validation_error(
                "recurrence result exceeds the "
                f"{MAX_COMBINATORICS_RESULT_RATIONAL_DIGITS}-digit bound"
            )
        _require_bounded_fraction(
            value,
            max_digits=MAX_COMBINATORICS_RESULT_RATIONAL_DIGITS,
            label="recurrence result",
        )
    _validate_result_inline_size(
        {
            "coefficient_convention": coefficient_convention,
            "determinism": "DETERMINISTIC",
            "exactness": "EXACT_RATIONAL",
            "replay_prefix": [_fraction_wire(value) for value in replay],
            "replay_scope_end": requested_indices[-1],
            "scope": scope,
            "values": [
                {"index": index, "value": _fraction_wire(replay[index])}
                for index in requested_indices
            ],
        }
    )


def _validate_p_recursive_result_budget(
    *,
    coefficient_polynomials: tuple[tuple[CanonicalRational, ...], ...],
    initial_values: tuple[CanonicalRational, ...],
    coefficient_convention: str,
    polynomial_convention: str,
    scope: str,
    requested_indices: tuple[int, ...],
) -> None:
    polynomials = tuple(
        tuple(value.as_fraction() for value in polynomial)
        for polynomial in coefficient_polynomials
    )
    order = len(polynomials) - 1

    def polynomial_value(polynomial: tuple[Fraction, ...], index: int) -> Fraction:
        return sum(
            (
                coefficient * index**power
                for power, coefficient in enumerate(polynomial)
            ),
            start=Fraction(),
        )

    end = requested_indices[-1]
    replay = [value.as_fraction() for value in initial_values[: end + 1]]
    requested_index_set = set(requested_indices)
    minimum_size = 1_024 + sum(
        _minimum_fraction_wire_bytes(value) * (1 + (index in requested_index_set))
        for index, value in enumerate(replay)
    )
    residuals: list[tuple[int, Fraction]] = []
    while len(replay) <= end:
        index = len(replay)
        coefficients = tuple(
            polynomial_value(polynomial, index) for polynomial in polynomials
        )
        if coefficients[0] == 0:
            raise _recurrence_validation_error(
                f"leading coefficient polynomial vanishes at index {index}"
            )
        next_value = (
            -sum(
                (
                    coefficients[offset] * replay[index - offset]
                    for offset in range(1, order + 1)
                ),
                start=Fraction(),
            )
            / coefficients[0]
        )
        if any(
            _lower_decimal_digits(component) > MAX_COMBINATORICS_RESULT_RATIONAL_DIGITS
            for component in (next_value.numerator, next_value.denominator)
        ):
            raise _recurrence_validation_error(
                "polynomial-coefficient recurrence result exceeds the "
                f"{MAX_COMBINATORICS_RESULT_RATIONAL_DIGITS}-digit bound"
            )
        _require_bounded_fraction(
            next_value,
            max_digits=MAX_COMBINATORICS_RESULT_RATIONAL_DIGITS,
            label="polynomial-coefficient recurrence result",
        )
        minimum_size += _minimum_fraction_wire_bytes(next_value) * (
            1 + (index in requested_index_set)
        )
        minimum_size += 32
        if minimum_size > MAX_COMBINATORICS_RESULT_ARTIFACT_BYTES:
            raise _recurrence_validation_error(
                "the exact combinatorics result exceeds the bounded result limit"
            )
        replay.append(next_value)
        residuals.append(
            (
                index,
                sum(
                    (
                        coefficients[offset] * replay[index - offset]
                        for offset in range(order + 1)
                    ),
                    start=Fraction(),
                ),
            )
        )
    _validate_result_inline_size(
        {
            "coefficient_convention": coefficient_convention,
            "polynomial_convention": polynomial_convention,
            "determinism": "DETERMINISTIC",
            "exactness": "EXACT_RATIONAL",
            "recurrence_order": order,
            "replay_prefix": [_fraction_wire(value) for value in replay],
            "residuals": [
                {"index": index, "value": _fraction_wire(value)}
                for index, value in residuals
            ],
            "replay_scope_end": end,
            "scope": scope,
            "values": [
                {"index": index, "value": _fraction_wire(replay[index])}
                for index in requested_indices
            ],
        }
    )


def _validate_series_result_budget(
    *,
    numerator: tuple[CanonicalRational, ...],
    denominator: tuple[CanonicalRational, ...],
    coefficient_convention: str,
    expansion_point: str,
    truncation_order: int,
) -> None:
    numerator_values = tuple(value.as_fraction() for value in numerator)
    denominator_values = tuple(value.as_fraction() for value in denominator)
    coefficients: list[Fraction] = []
    for degree in range(truncation_order):
        numerator_coefficient = (
            numerator_values[degree] if degree < len(numerator_values) else Fraction()
        )
        known = sum(
            (
                denominator_values[offset] * coefficients[degree - offset]
                for offset in range(1, min(degree, len(denominator_values) - 1) + 1)
            ),
            start=Fraction(),
        )
        coefficient = (numerator_coefficient - known) / denominator_values[0]
        if any(
            _lower_decimal_digits(component) > MAX_COMBINATORICS_RESULT_RATIONAL_DIGITS
            for component in (coefficient.numerator, coefficient.denominator)
        ):
            raise _recurrence_validation_error(
                "series coefficient exceeds the "
                f"{MAX_COMBINATORICS_RESULT_RATIONAL_DIGITS}-digit bound"
            )
        _require_bounded_fraction(
            coefficient,
            max_digits=MAX_COMBINATORICS_RESULT_RATIONAL_DIGITS,
            label="series coefficient",
        )
        coefficients.append(coefficient)
    minimum_size = sum(_minimum_fraction_wire_bytes(value) for value in coefficients)
    minimum_size += truncation_order * _minimum_fraction_wire_bytes(Fraction())
    if minimum_size + 1_024 > MAX_COMBINATORICS_RESULT_ARTIFACT_BYTES:
        raise _recurrence_validation_error(
            "the exact combinatorics result exceeds the bounded result limit"
        )
    _validate_result_inline_size(
        {
            "coefficient_convention": coefficient_convention,
            "coefficients": [_fraction_wire(value) for value in coefficients],
            "determinism": "DETERMINISTIC",
            "exactness": "EXACT_RATIONAL",
            "expansion_point": expansion_point,
            "residual_coefficients": [_fraction_wire(Fraction())] * truncation_order,
            "residual_congruence": (
                "DENOMINATOR_TIMES_SERIES_MINUS_NUMERATOR_IS_ZERO_MOD_X_TO_ORDER"
            ),
            "truncation_order": truncation_order,
        }
    )


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

    n: StrictInt = Field(ge=0, le=10_000)
    f_n: CanonicalInteger
    f_n_plus_one: CanonicalInteger


class FibonacciPairRequest(StrictModel):
    n: StrictInt = Field(ge=0, le=10_000)


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
        requested_indices = (
            tuple(range(self.term_count))
            if self.scope == "PREFIX" and self.term_count is not None
            else self.indices
        )
        _validate_recurrence_result_budget(
            coefficients=self.coefficients,
            initial_values=self.initial_values,
            coefficient_convention=self.coefficient_convention,
            scope=self.scope,
            requested_indices=requested_indices,
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
    replay_prefix: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_LINEAR_RECURRENCE_INDEX + 1,
    )
    replay_scope_end: StrictInt = Field(ge=0, le=MAX_LINEAR_RECURRENCE_INDEX)
    exactness: Literal["EXACT_RATIONAL"] = "EXACT_RATIONAL"
    determinism: Literal["DETERMINISTIC"] = "DETERMINISTIC"

    @model_validator(mode="after")
    def require_complete_replay_prefix(self) -> Self:
        if len(self.replay_prefix) != self.replay_scope_end + 1:
            raise _recurrence_validation_error(
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
            raise _recurrence_validation_error(
                "result indices must be strictly increasing"
            )
        if indices[-1] != self.replay_scope_end:
            raise _recurrence_validation_error(
                "the greatest requested index must bind replay_scope_end"
            )
        if any(item.value != self.replay_prefix[item.index] for item in self.values):
            raise _recurrence_validation_error(
                "indexed values must match the recurrence replay prefix"
            )
        if self.scope == "PREFIX" and indices != tuple(
            range(self.replay_scope_end + 1)
        ):
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
        for polynomial in self.coefficient_polynomials:
            if (
                not polynomial
                or len(polynomial) > MAX_P_RECURSIVE_POLYNOMIAL_DEGREE + 1
            ):
                raise _recurrence_validation_error(
                    "coefficient polynomial degree is outside the bound"
                )
            _require_canonical_polynomial(
                polynomial, label="recurrence polynomial coefficient"
            )
        for value in self.initial_values:
            _require_bounded_rational(
                value,
                max_digits=MAX_COMBINATORICS_INPUT_RATIONAL_DIGITS,
                label="recurrence initial value",
            )
        if self.scope == "PREFIX":
            if self.term_count is None or self.indices:
                raise _recurrence_validation_error(
                    "PREFIX scope requires term_count and forbids indices"
                )
            requested = tuple(range(self.term_count))
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
            requested = self.indices
        _validate_p_recursive_result_budget(
            coefficient_polynomials=self.coefficient_polynomials,
            initial_values=self.initial_values,
            coefficient_convention=self.coefficient_convention,
            polynomial_convention=self.polynomial_convention,
            scope=self.scope,
            requested_indices=requested,
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
    replay_prefix: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_LINEAR_RECURRENCE_INDEX + 1
    )
    residuals: tuple[IndexedRationalValue, ...] = Field(
        max_length=MAX_LINEAR_RECURRENCE_INDEX + 1
    )
    replay_scope_end: StrictInt = Field(ge=0, le=MAX_LINEAR_RECURRENCE_INDEX)
    exactness: Literal["EXACT_RATIONAL"] = "EXACT_RATIONAL"
    determinism: Literal["DETERMINISTIC"] = "DETERMINISTIC"

    @model_validator(mode="after")
    def require_complete_replay(self) -> Self:
        if len(self.replay_prefix) != self.replay_scope_end + 1:
            raise _recurrence_validation_error(
                "replay_prefix must cover the complete bounded scope"
            )
        indices = tuple(item.index for item in self.values)
        if any(left >= right for left, right in pairwise(indices)):
            raise _recurrence_validation_error(
                "result indices must be strictly increasing"
            )
        if indices[-1] != self.replay_scope_end:
            raise _recurrence_validation_error(
                "the greatest requested index must bind replay_scope_end"
            )
        if any(item.value != self.replay_prefix[item.index] for item in self.values):
            raise _recurrence_validation_error(
                "indexed values must match the recurrence replay prefix"
            )
        if self.scope == "PREFIX" and indices != tuple(range(len(indices))):
            raise _recurrence_validation_error(
                "PREFIX results must contain consecutive indices from zero"
            )
        for value in self.replay_prefix:
            _require_bounded_rational(
                value,
                max_digits=MAX_COMBINATORICS_RESULT_RATIONAL_DIGITS,
                label="polynomial-coefficient recurrence replay value",
            )
        residual_indices = tuple(item.index for item in self.residuals)
        if residual_indices != tuple(
            range(self.recurrence_order, self.replay_scope_end + 1)
        ):
            raise _recurrence_validation_error(
                "residuals must cover every recurrence step through replay_scope_end"
            )
        if any(item.value.as_fraction() != 0 for item in self.residuals):
            raise _recurrence_validation_error(
                "every recurrence residual must be exactly zero"
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

    @model_validator(mode="after")
    def require_regular_canonical_input(self) -> Self:
        _require_canonical_polynomial(self.numerator, label="numerator coefficient")
        _require_canonical_polynomial(
            self.denominator,
            label="denominator coefficient",
        )
        if self.denominator[0].as_fraction() == 0:
            raise _recurrence_validation_error(
                "denominator constant coefficient must be nonzero"
            )
        _validate_series_result_budget(
            numerator=self.numerator,
            denominator=self.denominator,
            coefficient_convention=self.coefficient_convention,
            expansion_point=self.expansion_point,
            truncation_order=self.truncation_order,
        )
        return self


class RationalGeneratingFunctionCoefficientsResult(StrictModel):
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

    @model_validator(mode="after")
    def require_exact_finite_truncation(self) -> Self:
        if (
            len(self.coefficients) != self.truncation_order
            or len(self.residual_coefficients) != self.truncation_order
        ):
            raise _recurrence_validation_error(
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
            raise _recurrence_validation_error(
                "residual coefficients must vanish through the truncation"
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
