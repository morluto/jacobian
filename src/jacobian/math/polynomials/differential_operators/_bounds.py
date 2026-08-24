"""Pre-execution bounds for differential-operator application."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction

from jacobian._exact import require_bounded_rational
from jacobian.math.polynomials.differential_operators.values import (
    ConstantCoefficientDifferentialOperator,
)
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_EXPONENT,
    RationalPolynomial,
    require_polynomial_budget,
)

MAX_APPLICATION_ITERATIONS = 4_096
MAX_APPLICATION_INPUT_TERMS = 512
MAX_APPLICATION_INPUT_EXPONENT = 128
MAX_APPLICATION_INPUT_TOTAL_DEGREE = 128
MAX_APPLICATION_INPUT_COEFFICIENT_DIGITS = 256
MAX_APPLICATION_OPERATOR_COEFFICIENT_DIGITS = 256
MAX_APPLICATION_OUTPUT_TERMS = 4_096
MAX_APPLICATION_OUTPUT_COEFFICIENT_DIGITS = 32_768
MAX_APPLICATION_WORK_UNITS = 2_000_000
MAX_APPLICATION_ARITHMETIC_WORK_UNITS = 8_000_000
MAX_APPLICATION_RESULT_BYTES = 9 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ApplicationEnvelope:
    """Preflight quantities used by the admitted FLINT kernel."""

    guaranteed_zero: bool
    expanded_operator_terms: int
    candidate_output_terms: int


def _total_degree(polynomial: RationalPolynomial) -> int:
    return max(
        (sum(term.exponents) for term in polynomial.polynomial.terms),
        default=0,
    )


def _guaranteed_zero(
    polynomial: RationalPolynomial,
    operator: ConstantCoefficientDifferentialOperator,
    iterations: int,
) -> bool:
    if not polynomial.polynomial.terms:
        return True
    if iterations == 0:
        return False
    if not operator.terms:
        return True
    # Every composition of k operator terms has at least k times the per-axis
    # minimum order on each axis, so a strict per-axis excess over the source's
    # widest exponent annihilates every monomial without any cancellation.
    axes = range(len(polynomial.variables))
    exponent_bounds = [
        max(term.exponents[axis] for term in polynomial.polynomial.terms)
        for axis in axes
    ]
    minimum_axis_orders = [
        min(term.orders[axis] for term in operator.terms) for axis in axes
    ]
    if any(
        iterations * minimum_axis_orders[axis] > exponent_bounds[axis] for axis in axes
    ):
        return True
    minimum_total_order = min(sum(term.orders) for term in operator.terms)
    return minimum_total_order > 0 and (
        minimum_total_order * iterations > _total_degree(polynomial)
    )


def _is_identity_operator(
    operator: ConstantCoefficientDifferentialOperator,
) -> bool:
    """Recognize the multiplicative identity operator ``1``."""

    return (
        len(operator.terms) == 1
        and not any(operator.terms[0].orders)
        and operator.terms[0].coefficient.as_fraction() == 1
    )


def _bounded_multiset_count(term_count: int, iterations: int, limit: int) -> int:
    """Bound the support of a commuting operator power.

    A power of a ``term_count``-term operator has at most
    ``C(term_count + iterations - 1, iterations)`` aggregate multi-indices.
    """

    if iterations == 0 or term_count <= 1:
        return 1
    n = term_count + iterations - 1
    r = min(iterations, term_count - 1)
    result = 1
    for index in range(1, r + 1):
        result = result * (n - r + index) // index
        if result > limit:
            return limit + 1
    return result


def _operator_power_work(term_count: int, iterations: int) -> int:
    """Bound sparse term-pair products in the kernel's binary power schedule."""

    def support(exponent: int) -> int:
        return _bounded_multiset_count(
            term_count,
            exponent,
            MAX_APPLICATION_OUTPUT_TERMS,
        )

    work = 0
    powered_exponent = 0
    base_exponent = 1
    remaining = iterations
    while remaining:
        if remaining & 1:
            work += support(powered_exponent) * support(base_exponent)
            powered_exponent += base_exponent
        remaining >>= 1
        if remaining:
            work += support(base_exponent) ** 2
            base_exponent *= 2
    return work


def _operator_path_bit_bound(term_count: int, iterations: int) -> int:
    """Bound the bit length of ``term_count**iterations`` without expanding it."""

    if iterations == 0 or term_count <= 1:
        return 1
    return iterations * (term_count - 1).bit_length() + 1


def _common_denominator_height(values: Iterable[Fraction]) -> tuple[int, int]:
    fractions = tuple(values)
    if not fractions:
        return 1, 0
    denominator = 1
    for value in fractions:
        denominator = math.lcm(denominator, value.denominator)
    maximum_scaled_numerator = max(
        abs(value.numerator) * (denominator // value.denominator) for value in fractions
    )
    return denominator, maximum_scaled_numerator


def _power_bit_bound(value: int, exponent: int) -> int:
    if exponent == 0 or value <= 1:
        return 1
    return value.bit_length() * exponent


def _decimal_digits_from_bits(bits: int) -> int:
    # 30103 / 100000 is a strict upper rational approximation to log10(2).
    return max(1, (bits * 30_103 + 99_999) // 100_000)


def _max_coefficient_digits(polynomial: RationalPolynomial) -> int:
    return max(
        (
            max(len(term.coefficient.num.lstrip("-")), len(term.coefficient.den))
            for term in polynomial.polynomial.terms
        ),
        default=1,
    )


def _coefficient_digit_bound(
    polynomial: RationalPolynomial,
    operator: ConstantCoefficientDifferentialOperator,
    iterations: int,
    path_count_bits: int,
) -> int:
    if iterations == 0:
        return _max_coefficient_digits(polynomial)

    source_denominator, source_numerator = _common_denominator_height(
        term.coefficient.as_fraction() for term in polynomial.polynomial.terms
    )
    operator_denominator, operator_numerator = _common_denominator_height(
        term.coefficient.as_fraction() for term in operator.terms
    )
    degree = _total_degree(polynomial)
    maximum_operator_order = max(sum(term.orders) for term in operator.terms)
    derivative_order = min(degree, iterations * maximum_operator_order)
    falling_factor_bound = (
        1
        if derivative_order == 0
        else min(math.factorial(degree), degree**derivative_order)
    )

    numerator_bits = sum(
        (
            max(1, len(polynomial.polynomial.terms)).bit_length(),
            path_count_bits,
            max(1, source_numerator).bit_length(),
            _power_bit_bound(operator_numerator, iterations),
            max(1, falling_factor_bound).bit_length(),
        )
    )
    denominator_bits = source_denominator.bit_length()
    if operator_denominator > 1:
        denominator_bits += operator_denominator.bit_length() * iterations
    return max(
        _decimal_digits_from_bits(numerator_bits),
        _decimal_digits_from_bits(denominator_bits),
    )


def _require_result_size(
    polynomial: RationalPolynomial,
    operator: ConstantCoefficientDifferentialOperator,
    expected: RationalPolynomial | None,
    *,
    candidate_terms: int,
    coefficient_digits: int,
) -> None:
    def polynomial_bytes(value: RationalPolynomial) -> int:
        size = sum(len(variable) + 4 for variable in value.variables) + 256
        return size + sum(
            len(term.coefficient.num)
            + len(term.coefficient.den)
            + sum(len(str(exponent)) + 2 for exponent in term.exponents)
            + 96
            for term in value.polynomial.terms
        )

    def operator_bytes(value: ConstantCoefficientDifferentialOperator) -> int:
        size = sum(len(variable) + 4 for variable in value.variables) + 256
        return size + sum(
            len(term.coefficient.num)
            + len(term.coefficient.den)
            + sum(len(str(order)) + 2 for order in term.orders)
            + 96
            for term in value.terms
        )

    retained_bytes = polynomial_bytes(polynomial) + operator_bytes(operator)
    if expected is not None:
        retained_bytes += polynomial_bytes(expected)

    # Output monomials never exceed the source's exponents, so the source's
    # widest exponent bounds every output term's serialized exponent digits.
    exponent_digits = max(
        (
            max(len(str(exponent)) for exponent in term.exponents)
            for term in polynomial.polynomial.terms
        ),
        default=1,
    )
    per_output_term = (
        2 * coefficient_digits + len(polynomial.variables) * (exponent_digits + 2) + 96
    )
    output_axis_bytes = sum(len(variable) for variable in polynomial.variables) + 256
    estimated_bytes = (
        retained_bytes + candidate_terms * per_output_term + output_axis_bytes + 2_048
    )
    if estimated_bytes > MAX_APPLICATION_RESULT_BYTES:
        raise ValueError(
            "differential-operator result exceeds the aggregate serialized-output "
            f"budget of {MAX_APPLICATION_RESULT_BYTES} bytes"
        )


def _require_application_shape(
    polynomial: RationalPolynomial,
    operator: ConstantCoefficientDifferentialOperator,
    expected: RationalPolynomial | None,
    iterations: int,
) -> None:
    """Apply the request checks that every execution path exercises."""

    if isinstance(iterations, bool) or not isinstance(iterations, int):
        raise TypeError("differential-operator iterations must be an integer")
    if polynomial.variables != operator.variables:
        raise ValueError(
            "polynomial and differential operator must use the same ordered variables"
        )
    if expected is not None and expected.variables != polynomial.variables:
        raise ValueError(
            "expected polynomial must use the source polynomial's ordered variables"
        )
    if iterations < 0:
        raise ValueError("differential-operator iterations must be nonnegative")


def _require_expansion_operator(
    operator: ConstantCoefficientDifferentialOperator,
) -> None:
    """Bound the operator against the kernel's derivative-powering input regime."""

    for term in operator.terms:
        require_bounded_rational(
            term.coefficient,
            max_digits=MAX_APPLICATION_OPERATOR_COEFFICIENT_DIGITS,
            label="differential-operator coefficient",
        )


def _require_expansion_source(polynomial: RationalPolynomial) -> None:
    """Bound the source against the kernel's derivative-expansion input regime."""

    require_polynomial_budget(
        polynomial,
        maximum_terms=MAX_APPLICATION_INPUT_TERMS,
        maximum_exponent=MAX_APPLICATION_INPUT_EXPONENT,
        maximum_coefficient_digits=MAX_APPLICATION_INPUT_COEFFICIENT_DIGITS,
        label="differential-operator source polynomial",
    )
    if _total_degree(polynomial) > MAX_APPLICATION_INPUT_TOTAL_DEGREE:
        raise ValueError(
            "differential-operator source polynomial exceeds the total-degree budget"
        )


def _require_identity_output(
    polynomial: RationalPolynomial,
    expected: RationalPolynomial | None,
) -> None:
    """Admit copy results by the output budgets they actually exercise.

    With ``iterations == 0`` or the identity operator, the exact result is the
    source itself and no expansion runs, so admission follows the copied result
    (output support, digit, and aggregate-byte budgets) rather than the kernel
    input regime.
    """

    require_polynomial_budget(
        polynomial,
        maximum_terms=MAX_APPLICATION_OUTPUT_TERMS,
        maximum_exponent=MAX_POLYNOMIAL_EXPONENT,
        maximum_coefficient_digits=MAX_APPLICATION_OUTPUT_COEFFICIENT_DIGITS,
        label="differential-operator source polynomial",
    )
    if expected is not None:
        _require_expected_output(expected)


def _require_expected_output(expected: RationalPolynomial | None) -> None:
    """Bound the retained comparison value, which never enters the expansion.

    ``expected`` is only retained in the result and compared for exact
    equality, so admission follows the shared canonical representation plus
    the aggregate retained-byte budget rather than the kernel's input regime.
    """

    if expected is None:
        return
    require_polynomial_budget(
        expected,
        maximum_terms=MAX_APPLICATION_OUTPUT_TERMS,
        maximum_exponent=MAX_POLYNOMIAL_EXPONENT,
        maximum_coefficient_digits=MAX_APPLICATION_OUTPUT_COEFFICIENT_DIGITS,
        label="expected differential-operator output",
    )


def validate_application_envelope(
    polynomial: RationalPolynomial,
    operator: ConstantCoefficientDifferentialOperator,
    iterations: int,
    expected: RationalPolynomial | None,
) -> ApplicationEnvelope:
    """Validate the complete public domain before FLINT expansion."""

    _require_application_shape(polynomial, operator, expected, iterations)

    # Degenerate shortcuts establish their exact results without running the
    # kernel, so they are recognized before expansion-specific source limits
    # and admitted by the result and work they actually have. Positive powers
    # of the identity operator join the zero iterate: 1^k(f) = f for every k.
    guaranteed_zero = _guaranteed_zero(polynomial, operator, iterations)
    if guaranteed_zero:
        _require_result_size(
            polynomial,
            operator,
            expected,
            candidate_terms=0,
            coefficient_digits=1,
        )
        return ApplicationEnvelope(True, 0, 0)

    if iterations == 0 or _is_identity_operator(operator):
        _require_identity_output(polynomial, expected)
        _require_result_size(
            polynomial,
            operator,
            expected,
            candidate_terms=len(polynomial.polynomial.terms),
            coefficient_digits=_max_coefficient_digits(polynomial),
        )
        return ApplicationEnvelope(False, 1, len(polynomial.polynomial.terms))

    if iterations > MAX_APPLICATION_ITERATIONS:
        raise ValueError("differential-operator iterations exceed the operation limit")

    _require_expansion_operator(operator)
    _require_expansion_source(polynomial)
    _require_expected_output(expected)

    term_count = len(operator.terms)
    expanded_terms = _bounded_multiset_count(
        term_count,
        iterations,
        MAX_APPLICATION_OUTPUT_TERMS,
    )
    if expanded_terms > MAX_APPLICATION_OUTPUT_TERMS:
        raise ValueError(
            "differential-operator power exceeds the expanded-support budget"
        )
    candidate_terms = len(polynomial.polynomial.terms) * expanded_terms
    if candidate_terms > MAX_APPLICATION_OUTPUT_TERMS:
        raise ValueError(
            "differential-operator output exceeds the candidate-term budget"
        )

    coefficient_digits = _coefficient_digit_bound(
        polynomial,
        operator,
        iterations,
        _operator_path_bit_bound(term_count, iterations),
    )
    if coefficient_digits > MAX_APPLICATION_OUTPUT_COEFFICIENT_DIGITS:
        raise ValueError(
            "differential-operator output exceeds the coefficient-digit budget"
        )

    maximum_operator_order = max(
        (sum(term.orders) for term in operator.terms),
        default=0,
    )
    derivative_order = min(
        _total_degree(polynomial),
        iterations * maximum_operator_order,
    )
    power_work = _operator_power_work(term_count, iterations)
    derivative_work = (
        len(polynomial.polynomial.terms) * expanded_terms * (1 + derivative_order)
    )
    conversion_work = 2 * (
        len(polynomial.polynomial.terms) + term_count + candidate_terms
    )
    # Result validation replays the same defining relation, so the public path
    # pays for two complete applications rather than hiding replay work.
    work_units = 2 * (power_work + derivative_work) + conversion_work
    if work_units > MAX_APPLICATION_WORK_UNITS:
        raise ValueError(
            "differential-operator application exceeds the deterministic work budget"
        )
    arithmetic_work = work_units * max(1, (coefficient_digits + 255) // 256)
    if arithmetic_work > MAX_APPLICATION_ARITHMETIC_WORK_UNITS:
        raise ValueError(
            "differential-operator application exceeds the coefficient-arithmetic "
            "work budget"
        )
    _require_result_size(
        polynomial,
        operator,
        expected,
        candidate_terms=candidate_terms,
        coefficient_digits=coefficient_digits,
    )
    return ApplicationEnvelope(
        guaranteed_zero=False,
        expanded_operator_terms=expanded_terms,
        candidate_output_terms=candidate_terms,
    )


__all__ = [
    "MAX_APPLICATION_INPUT_COEFFICIENT_DIGITS",
    "MAX_APPLICATION_INPUT_EXPONENT",
    "MAX_APPLICATION_INPUT_TERMS",
    "MAX_APPLICATION_INPUT_TOTAL_DEGREE",
    "MAX_APPLICATION_ITERATIONS",
    "MAX_APPLICATION_OPERATOR_COEFFICIENT_DIGITS",
    "MAX_APPLICATION_OUTPUT_COEFFICIENT_DIGITS",
    "MAX_APPLICATION_OUTPUT_TERMS",
    "ApplicationEnvelope",
    "validate_application_envelope",
]
