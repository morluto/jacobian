"""Native admission and prepared kernels for exact recurrence operations."""

from __future__ import annotations

import math
from collections.abc import Callable
from fractions import Fraction

from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics._recurrence_models import (
    MAX_COMBINATORICS_INPUT_RATIONAL_DIGITS,
    MAX_COMBINATORICS_RESULT_ARTIFACT_BYTES,
    MAX_COMBINATORICS_RESULT_RATIONAL_DIGITS,
    MAX_P_RECURSIVE_POLYNOMIAL_DEGREE,
    _require_bounded_rational,
    _require_canonical_polynomial,
    _validate_result_inline_size,
)

_LOG10_2 = math.log10(2)
_FRACTION_WIRE_FIXED_BYTES = 20
_RESULT_WIRE_FIXED_BYTES = 1_024


def _run_admission(
    admission: Callable[[], None], *, location: tuple[str | int, ...]
) -> None:
    """Translate owner helper failures into the typed operation boundary."""

    try:
        admission()
    except OperationDomainValidationError:
        raise
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=location, code=exc.type, message=exc.message()
        ) from exc
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=location, code="combinatorics.admission", message=str(exc)
        ) from exc


def _admit_bounded_input(
    value: CanonicalRational,
    *,
    label: str,
    location: tuple[str | int, ...],
) -> None:
    """Apply an input rational bound with a field-specific diagnostic."""

    try:
        _require_bounded_rational(
            value,
            max_digits=MAX_COMBINATORICS_INPUT_RATIONAL_DIGITS,
            label=label,
        )
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=location, code=exc.type, message=exc.message()
        ) from exc
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=location, code="combinatorics.admission", message=str(exc)
        ) from exc


def _lower_decimal_digits(value: int) -> int:
    if value == 0:
        return 1
    return math.floor((abs(value).bit_length() - 1) * _LOG10_2) + 1


def _fraction_wire(value: Fraction) -> dict[str, str]:
    return {
        "num": format_canonical_integer(value.numerator),
        "den": format_canonical_integer(value.denominator),
    }


def _minimum_fraction_wire_bytes(value: Fraction) -> int:
    return (
        _lower_decimal_digits(value.numerator)
        + _lower_decimal_digits(value.denominator)
        + _FRACTION_WIRE_FIXED_BYTES
    )


def _require_bounded_fraction(
    value: Fraction, *, label: str, location: tuple[str | int, ...]
) -> None:
    if any(
        _lower_decimal_digits(component) > MAX_COMBINATORICS_RESULT_RATIONAL_DIGITS
        for component in (value.numerator, value.denominator)
    ):
        raise OperationDomainValidationError(
            location=location,
            code="combinatorics.rational_bound",
            message=(
                f"{label} exceeds the "
                f"{MAX_COMBINATORICS_RESULT_RATIONAL_DIGITS}-digit bound"
            ),
        )


def _admit_linear_recurrence(
    *,
    coefficients: tuple[CanonicalRational, ...],
    initial_values: tuple[CanonicalRational, ...],
    coefficient_convention: str,
    scope: str,
    requested_indices: tuple[int, ...],
) -> tuple[Fraction, ...]:
    """Prepare the recurrence prefix while checking its exact result envelope."""

    for label, values in (
        ("recurrence coefficient", coefficients),
        ("recurrence initial value", initial_values),
    ):
        for index, rational_value in enumerate(values):
            _admit_bounded_input(
                rational_value,
                label=label,
                location=(
                    "coefficients"
                    if label == "recurrence coefficient"
                    else "initial_values",
                    index,
                ),
            )
    prefix = [
        value.as_fraction() for value in initial_values[: requested_indices[-1] + 1]
    ]
    coefficient_values = tuple(value.as_fraction() for value in coefficients)
    while len(prefix) <= requested_indices[-1]:
        prefix.append(
            sum(
                (
                    coefficient * prefix[len(prefix) - offset]
                    for offset, coefficient in enumerate(coefficient_values, start=1)
                ),
                start=Fraction(),
            )
        )
    minimum_size = sum(
        _minimum_fraction_wire_bytes(prefix[index]) for index in requested_indices
    )
    if (
        minimum_size + _RESULT_WIRE_FIXED_BYTES
        > MAX_COMBINATORICS_RESULT_ARTIFACT_BYTES
    ):
        raise OperationDomainValidationError(
            location=("values",),
            code="combinatorics.result_bound",
            message="the exact combinatorics result exceeds the bounded result limit",
        )
    for index, fraction_value in enumerate(prefix):
        _require_bounded_fraction(
            fraction_value,
            label="recurrence result",
            location=("values", index),
        )
    _run_admission(
        lambda: _validate_result_inline_size(
            {
                "coefficient_convention": coefficient_convention,
                "scope": scope,
                "values": [
                    {"index": index, "value": _fraction_wire(prefix[index])}
                    for index in requested_indices
                ],
            }
        ),
        location=("values",),
    )
    return tuple(prefix)


def _admit_p_recursive_recurrence(
    *,
    coefficient_polynomials: tuple[tuple[CanonicalRational, ...], ...],
    initial_values: tuple[CanonicalRational, ...],
    coefficient_convention: str,
    polynomial_convention: str,
    scope: str,
    requested_indices: tuple[int, ...],
) -> tuple[Fraction, ...]:
    """Prepare a P-recursive prefix and check the projected result envelope."""

    for polynomial_index, polynomial in enumerate(coefficient_polynomials):
        if not polynomial or len(polynomial) > MAX_P_RECURSIVE_POLYNOMIAL_DEGREE + 1:
            raise OperationDomainValidationError(
                location=("coefficient_polynomials", polynomial_index),
                code="combinatorics.polynomial_invariant",
                message="coefficient polynomial degree is outside the bound",
            )
        try:
            _require_canonical_polynomial(
                polynomial, label="recurrence polynomial coefficient"
            )
        except PydanticCustomError as exc:
            raise OperationDomainValidationError(
                location=("coefficient_polynomials", polynomial_index),
                code=exc.type,
                message=exc.message(),
            ) from exc
    for index, value in enumerate(initial_values):
        _admit_bounded_input(
            value,
            label="recurrence initial value",
            location=("initial_values", index),
        )
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
    prefix = [value.as_fraction() for value in initial_values[: end + 1]]
    requested_index_set = set(requested_indices)
    minimum_size = _RESULT_WIRE_FIXED_BYTES + sum(
        _minimum_fraction_wire_bytes(value)
        for index, value in enumerate(prefix)
        if index in requested_index_set
    )
    while len(prefix) <= end:
        index = len(prefix)
        coefficients = tuple(
            polynomial_value(polynomial, index) for polynomial in polynomials
        )
        if coefficients[0] == 0:
            raise OperationDomainValidationError(
                location=("coefficient_polynomials", 0),
                code="combinatorics.recurrence_invariant",
                message=f"leading coefficient polynomial vanishes at index {index}",
            )
        next_value = (
            -sum(
                (
                    coefficients[offset] * prefix[index - offset]
                    for offset in range(1, order + 1)
                ),
                start=Fraction(),
            )
            / coefficients[0]
        )
        _require_bounded_fraction(
            next_value,
            label="polynomial-coefficient recurrence result",
            location=("values", index),
        )
        if index in requested_index_set:
            minimum_size += _minimum_fraction_wire_bytes(next_value)
        if minimum_size > MAX_COMBINATORICS_RESULT_ARTIFACT_BYTES:
            raise OperationDomainValidationError(
                location=("values",),
                code="combinatorics.result_bound",
                message="the exact combinatorics result exceeds the bounded result limit",
            )
        prefix.append(next_value)
    _run_admission(
        lambda: _validate_result_inline_size(
            {
                "coefficient_convention": coefficient_convention,
                "polynomial_convention": polynomial_convention,
                "recurrence_order": order,
                "scope": scope,
                "values": [
                    {"index": index, "value": _fraction_wire(prefix[index])}
                    for index in requested_indices
                ],
            }
        ),
        location=("values",),
    )
    return tuple(prefix)


def _admit_series(
    *,
    numerator: tuple[CanonicalRational, ...],
    denominator: tuple[CanonicalRational, ...],
    coefficient_convention: str,
    expansion_point: str,
    truncation_order: int,
) -> tuple[Fraction, ...]:
    """Prepare a rational-series prefix while checking its exact result envelope."""

    _run_admission(
        lambda: _require_canonical_polynomial(numerator, label="numerator coefficient"),
        location=("numerator",),
    )
    _run_admission(
        lambda: _require_canonical_polynomial(
            denominator, label="denominator coefficient"
        ),
        location=("denominator",),
    )
    if denominator[0].as_fraction() == 0:
        raise OperationDomainValidationError(
            location=("denominator", 0),
            code="combinatorics.recurrence_invariant",
            message="denominator constant coefficient must be nonzero",
        )
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
        _require_bounded_fraction(
            coefficient,
            label="series coefficient",
            location=("coefficients", degree),
        )
        coefficients.append(coefficient)
    minimum_size = sum(_minimum_fraction_wire_bytes(value) for value in coefficients)
    minimum_size += truncation_order * _minimum_fraction_wire_bytes(Fraction())
    if (
        minimum_size + _RESULT_WIRE_FIXED_BYTES
        > MAX_COMBINATORICS_RESULT_ARTIFACT_BYTES
    ):
        raise OperationDomainValidationError(
            location=("coefficients",),
            code="combinatorics.result_bound",
            message="the exact combinatorics result exceeds the bounded result limit",
        )
    _run_admission(
        lambda: _validate_result_inline_size(
            {
                "coefficient_convention": coefficient_convention,
                "coefficients": [_fraction_wire(value) for value in coefficients],
                "expansion_point": expansion_point,
                "residual_coefficients": [_fraction_wire(Fraction())]
                * truncation_order,
                "residual_congruence": (
                    "DENOMINATOR_TIMES_SERIES_MINUS_NUMERATOR_IS_ZERO_MOD_X_TO_ORDER"
                ),
                "truncation_order": truncation_order,
            }
        ),
        location=("coefficients",),
    )
    return tuple(coefficients)


__all__ = [
    "_admit_linear_recurrence",
    "_admit_p_recursive_recurrence",
    "_admit_series",
]
