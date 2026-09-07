"""Native admission and prepared kernels for exact recurrence operations."""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction

from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics._recurrence_models import (
    MAX_COMBINATORICS_INPUT_RATIONAL_DIGITS,
    MAX_COMBINATORICS_RESULT_RATIONAL_DIGITS,
    MAX_P_RECURSIVE_POLYNOMIAL_DEGREE,
    MAX_RATIONAL_SERIES_WORK_UNITS,
    _require_bounded_rational,
    _require_canonical_polynomial,
)


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
    """Return the exact decimal width without converting through ``str``.

    The bit-length estimate only selects a nearby power of ten.  Integer
    comparisons correct that estimate, so powers of ten at the result bound
    are not underestimated and values just below them remain admitted.
    """

    if value == 0:
        return 1
    magnitude = abs(value)
    digits = ((magnitude.bit_length() - 1) * 30103) // 100000 + 1
    lower_power = 10 ** (digits - 1)
    while magnitude < lower_power:
        digits -= 1
        lower_power //= 10
    while magnitude >= lower_power * 10:
        lower_power *= 10
        digits += 1
    return digits


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


def _rational_series_work_units(denominator_degree: int, truncation_order: int) -> int:
    """Return the exact recurrence work charged by a rational-series prefix."""

    ramp = min(max(0, truncation_order - 1), denominator_degree)
    recurrence_products = ramp * (ramp + 1) // 2
    recurrence_products += (
        max(0, truncation_order - denominator_degree - 1) * denominator_degree
    )
    return truncation_order + recurrence_products


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
    for index, fraction_value in enumerate(prefix):
        _require_bounded_fraction(
            fraction_value,
            label="recurrence result",
            location=("values", index),
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
        prefix.append(next_value)
    return tuple(prefix)


def _admit_series(
    *,
    numerator: tuple[CanonicalRational, ...],
    denominator: tuple[CanonicalRational, ...],
    coefficient_convention: str,
    expansion_point: int,
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
    denominator_degree = len(denominator_values) - 1
    work_units = _rational_series_work_units(denominator_degree, truncation_order)
    if work_units > MAX_RATIONAL_SERIES_WORK_UNITS:
        raise OperationDomainValidationError(
            location=("truncation_order",),
            code="combinatorics.work_bound",
            message="rational-series recurrence exceeds the exact work bound",
        )
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
    return tuple(coefficients)


__all__ = [
    "_admit_linear_recurrence",
    "_admit_p_recursive_recurrence",
    "_admit_series",
    "_lower_decimal_digits",
    "_rational_series_work_units",
]
