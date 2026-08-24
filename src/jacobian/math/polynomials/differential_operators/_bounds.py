"""Pre-execution bounds for differential-operator application."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction

from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    require_bounded_rational,
)
from jacobian.math.polynomials.differential_operators.values import (
    ConstantCoefficientDifferentialOperator,
)
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_EXPONENT,
    MAX_POLYNOMIAL_TERMS,
    RationalPolynomial,
    require_polynomial_budget,
)

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
    rescale_only: bool = False


def _rescale_only(
    polynomial: RationalPolynomial,
    operator: ConstantCoefficientDifferentialOperator,
) -> bool:
    """Recognize requests whose only acting aggregate is the identity order.

    Every nonzero-order operator term must annihilate every source monomial,
    so the powered operator's action collapses to rescaling the source by
    the zero-order coefficient raised to the iteration count.
    """

    if not operator.terms or not polynomial.polynomial.terms:
        return False
    return all(
        any(
            order > exponent
            for order, exponent in zip(term.orders, monomial.exponents, strict=True)
        )
        for term in operator.terms
        if any(term.orders)
        for monomial in polynomial.polynomial.terms
    )


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


def _is_scalar_operator(
    operator: ConstantCoefficientDifferentialOperator,
) -> bool:
    """Recognize one-term zeroth-order operators, whose action is pure scaling."""

    return len(operator.terms) == 1 and not any(operator.terms[0].orders)


def _is_signed_unit_scalar(
    operator: ConstantCoefficientDifferentialOperator,
) -> bool:
    """Recognize scalar operators whose power is a pure sign, ``(±1)^k``."""

    return (
        _is_scalar_operator(operator)
        and abs(operator.terms[0].coefficient.as_fraction()) == 1
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


def _bounded_box_count(axis_counts: Iterable[int], limit: int) -> int:
    """Bound the support of a power inside a per-axis order box.

    Distinct aggregate multi-indices of a power have per-axis sums between
    ``0`` and ``iterations * maximum_axis_order``, so the product of the
    per-axis counts bounds the distinct support even when multiset order
    sums collide.
    """

    result = 1
    for count in axis_counts:
        result *= count
        if result > limit:
            return limit + 1
    return result


def _powered_support_bound(
    term_count: int,
    iterations: int,
    maximum_axis_orders: tuple[int, ...],
    limit: int,
) -> int:
    """Bound the distinct aggregate multi-indices of a powered operator."""

    return min(
        _bounded_multiset_count(term_count, iterations, limit),
        _bounded_box_count(
            (iterations * order + 1 for order in maximum_axis_orders),
            limit,
        ),
    )


ENUMERATION_WORK_CAP = 250_000


def _distinct_powered_orders(
    operator: ConstantCoefficientDifferentialOperator,
    iterations: int,
    limit: int,
) -> tuple[tuple[int, ...], ...] | None:
    """Enumerate the distinct aggregate multi-indices of a powered operator.

    Returns ``None`` when enumeration exceeds ``limit`` distinct sums or the
    work cap; callers then fall back to the analytic support bound.
    """

    if iterations == 0:
        return ()
    terms = tuple(term.orders for term in operator.terms)
    if not terms:
        return ()
    current = {tuple(orders) for orders in terms}
    work = len(current)
    for _ in range(iterations - 1):
        expanded: set[tuple[int, ...]] = set()
        exhausted = False
        for existing in current:
            for orders in terms:
                expanded.add(tuple(map(sum, zip(existing, orders, strict=True))))
                work += 1
                if len(expanded) > limit or work > ENUMERATION_WORK_CAP:
                    exhausted = True
                    break
            if exhausted:
                break
        if exhausted:
            return None
        current = expanded
    return tuple(sorted(current))


def _operator_power_work(
    term_count: int,
    iterations: int,
    maximum_axis_orders: tuple[int, ...],
) -> int:
    """Bound sparse term-pair products in the kernel's binary power schedule."""

    def support(exponent: int) -> int:
        return min(
            _bounded_multiset_count(term_count, exponent, MAX_APPLICATION_OUTPUT_TERMS),
            _bounded_box_count(
                (exponent * order + 1 for order in maximum_axis_orders),
                MAX_APPLICATION_OUTPUT_TERMS,
            ),
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


def _multiplier_bit_bound(value: int, exponent: int) -> int:
    """Bound the bits a multiplication by ``value**exponent`` can add.

    ``value <= 2 ** (value - 1).bit_length()`` for every positive integer, so
    multiplying by ``value`` adds at most ``(value - 1).bit_length()`` bits and
    unit factors add none.
    """

    if exponent == 0 or value <= 1:
        return 0
    return exponent * (value - 1).bit_length()


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


_OVERFLOW_BITS = 1 << 40


def _per_exponent_height_bits(
    polynomial: RationalPolynomial,
    operator: ConstantCoefficientDifferentialOperator,
    iterations: int,
    powered_orders: tuple[tuple[int, ...], ...],
) -> tuple[int, int] | None:
    """Bound output coefficient heights per colliding exponent class.

    Contributions merge only when differentiated source monomials land on the
    same output exponent, so denominators are combined per exponent class via
    their exact least common multiple instead of one global source LCM.
    Returns ``None`` when the accounting itself would overflow its cap, in
    which case the caller falls back to the coarser global bound.
    """

    cap = 64 * MAX_APPLICATION_OUTPUT_COEFFICIENT_DIGITS
    op_denominator, op_numerator = _common_denominator_height(
        term.coefficient.as_fraction() for term in operator.terms
    )
    op_numerator_bits = _multiplier_bit_bound(op_numerator, iterations)
    op_denominator_bits = _multiplier_bit_bound(op_denominator, iterations)
    multiplicity_bits = _multiplier_bit_bound(len(operator.terms), iterations)
    classes: dict[tuple[int, ...], tuple[int, int, int]] = {}
    for orders in powered_orders:
        for source_term in polynomial.polynomial.terms:
            if any(
                order > exponent
                for order, exponent in zip(orders, source_term.exponents, strict=True)
            ):
                continue
            target = tuple(
                exponent - order
                for exponent, order in zip(source_term.exponents, orders, strict=True)
            )
            falling_bits = sum(
                _multiplier_bit_bound(exponent, order)
                for exponent, order in zip(source_term.exponents, orders, strict=True)
            )
            fraction = source_term.coefficient.as_fraction()
            entry = classes.get(target)
            if entry is None:
                if (
                    fraction.denominator.bit_length() > cap
                    or abs(fraction.numerator).bit_length() > cap
                ):
                    return None
                classes[target] = (
                    fraction.denominator,
                    abs(fraction.numerator),
                    falling_bits,
                )
                continue
            lcm, numerator_sum, max_falling = entry
            new_lcm = math.lcm(lcm, fraction.denominator)
            scaled_existing = numerator_sum * (new_lcm // lcm)
            scaled_new = abs(fraction.numerator) * (new_lcm // fraction.denominator)
            merged = scaled_existing + scaled_new
            if new_lcm.bit_length() > cap or merged.bit_length() > cap:
                return None
            classes[target] = (
                new_lcm,
                merged,
                max(max_falling, falling_bits),
            )
    worst_numerator_bits = 0
    worst_denominator_bits = 0
    for lcm, numerator_sum, max_falling in classes.values():
        denominator_bits = lcm.bit_length() + op_denominator_bits
        numerator_bits = (
            numerator_sum.bit_length()
            + op_numerator_bits
            + multiplicity_bits
            + max_falling
        )
        worst_numerator_bits = max(worst_numerator_bits, numerator_bits)
        worst_denominator_bits = max(worst_denominator_bits, denominator_bits)
    return worst_numerator_bits, worst_denominator_bits


def _coefficient_digit_bound(
    polynomial: RationalPolynomial,
    operator: ConstantCoefficientDifferentialOperator,
    iterations: int,
    candidate_terms: int,
    powered_orders: tuple[tuple[int, ...], ...] | None,
) -> int:
    if iterations == 0:
        return _max_coefficient_digits(polynomial)

    operator_denominator, operator_numerator = _common_denominator_height(
        term.coefficient.as_fraction() for term in operator.terms
    )
    degree = _total_degree(polynomial)
    maximum_operator_order = max(sum(term.orders) for term in operator.terms)
    derivative_order = min(degree, iterations * maximum_operator_order)
    # Every per-axis falling factorial (e)_d satisfies (e)_d <= e^d and every
    # per-axis exponent e is at most the total degree, so the derivative
    # multiplier adds at most derivative_order * log2(degree) bits; the tight
    # multiplier bound above keeps unit factors - a degree-one source, a unit
    # operator coefficient, or a single candidate path - from adding any bit.
    falling_factor_bits = _multiplier_bit_bound(degree, derivative_order)

    # One aggregate multi-index receives a contribution from every ordered
    # term sequence of the power that produces it, so the coefficient bound
    # carries the multinomial path multiplicity, at most term_count **
    # iterations sequences.
    shared_numerator_bits = (
        _multiplier_bit_bound(candidate_terms, 1)
        + _multiplier_bit_bound(len(operator.terms), iterations)
        + _multiplier_bit_bound(operator_numerator, iterations)
        + falling_factor_bits
    )
    shared_denominator_bits = _multiplier_bit_bound(operator_denominator, iterations)

    if powered_orders is not None:
        per_exponent = _per_exponent_height_bits(
            polynomial,
            operator,
            iterations,
            powered_orders,
        )
        if per_exponent is not None:
            return max(
                _decimal_digits_from_bits(per_exponent[0]),
                _decimal_digits_from_bits(per_exponent[1]),
            )

    source_denominator, source_numerator = _common_denominator_height(
        term.coefficient.as_fraction() for term in polynomial.polynomial.terms
    )
    numerator_bits = source_numerator.bit_length() + shared_numerator_bits
    denominator_bits = source_denominator.bit_length() + shared_denominator_bits
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
    """Bound the operator against the kernel's derivative-powering input regime.

    Operator coefficient growth is derived from the coefficients' actual
    heights downstream; this check enforces only the shared canonical
    rational representation.
    """

    for term in operator.terms:
        require_bounded_rational(
            term.coefficient,
            max_digits=MAX_CANONICAL_RATIONAL_DIGITS,
            label="differential-operator coefficient",
        )


def _require_expansion_source(polynomial: RationalPolynomial) -> None:
    """Bound the source against the kernel's derivative-expansion input regime.

    Derivative work, coefficient growth, exact output support, and serialized
    size are derived from the source's actual support, exponents, and
    coefficients downstream; this check enforces only the shared canonical
    polynomial representation.
    """

    require_polynomial_budget(
        polynomial,
        maximum_terms=MAX_POLYNOMIAL_TERMS,
        maximum_exponent=MAX_POLYNOMIAL_EXPONENT,
        maximum_coefficient_digits=MAX_CANONICAL_RATIONAL_DIGITS,
        label="differential-operator source polynomial",
    )


def _require_nonexpanding_output(
    polynomial: RationalPolynomial,
    expected: RationalPolynomial | None,
) -> None:
    """Admit copy and pure-rescale results by the output budgets they exercise.

    With ``iterations == 0``, the identity operator, or a one-term zeroth-order
    operator, the exact result is the source up to rational scaling and no
    expansion runs, so admission follows the copied result (output support,
    digit, and aggregate-byte budgets) rather than the kernel input regime.
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


def _expansion_support_candidates(
    polynomial: RationalPolynomial,
    operator: ConstantCoefficientDifferentialOperator,
    iterations: int,
) -> tuple[int, int, tuple[tuple[int, ...], ...] | None]:
    """Bound the powered support and surviving output paths of one expansion."""

    term_count = len(operator.terms)
    maximum_axis_orders = tuple(
        max(term.orders[axis] for term in operator.terms)
        for axis in range(len(polynomial.variables))
    )
    powered_orders = _distinct_powered_orders(
        operator,
        iterations,
        MAX_APPLICATION_OUTPUT_TERMS,
    )
    if powered_orders is not None:
        expanded_terms = len(powered_orders)
    else:
        expanded_terms = _powered_support_bound(
            term_count,
            iterations,
            maximum_axis_orders,
            MAX_APPLICATION_OUTPUT_TERMS,
        )
    if expanded_terms > MAX_APPLICATION_OUTPUT_TERMS:
        raise ValueError(
            "differential-operator power exceeds the expanded-support budget"
        )
    # A powered aggregate acts on a source monomial only when every axis
    # order stays within that monomial's exponents, so the candidate support
    # counts acting aggregates per source monomial instead of treating
    # annihilating powered terms as surviving output paths.
    if powered_orders is not None:
        candidate_terms = 0
        for source_term in polynomial.polynomial.terms:
            candidate_terms += sum(
                1
                for orders in powered_orders
                if all(
                    order <= exponent
                    for order, exponent in zip(
                        orders, source_term.exponents, strict=True
                    )
                )
            )
            if candidate_terms > MAX_APPLICATION_OUTPUT_TERMS:
                raise ValueError(
                    "differential-operator output exceeds the candidate-term budget"
                )
        return expanded_terms, candidate_terms, powered_orders
    maximum_exponents = tuple(
        max(term.exponents[axis] for term in polynomial.polynomial.terms)
        for axis in range(len(polynomial.variables))
    )
    acting_terms = min(
        expanded_terms,
        _bounded_box_count(
            (
                min(exponent, iterations * order) + 1
                for exponent, order in zip(
                    maximum_exponents,
                    maximum_axis_orders,
                    strict=True,
                )
            ),
            MAX_APPLICATION_OUTPUT_TERMS,
        ),
    )
    candidate_terms = len(polynomial.polynomial.terms) * acting_terms
    if candidate_terms > MAX_APPLICATION_OUTPUT_TERMS:
        raise ValueError(
            "differential-operator output exceeds the candidate-term budget"
        )
    return expanded_terms, candidate_terms, powered_orders


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
        _require_nonexpanding_output(polynomial, expected)
        _require_result_size(
            polynomial,
            operator,
            expected,
            candidate_terms=len(polynomial.polynomial.terms),
            coefficient_digits=_max_coefficient_digits(polynomial),
        )
        return ApplicationEnvelope(False, 1, len(polynomial.polynomial.terms))

    # A one-term zeroth-order operator only rescales existing coefficients:
    # D^k(f) = c^k * f never expands derivative support, so its admission
    # follows the scale-only result's support, coefficient growth, work, and
    # size budgets rather than the derivative kernel's expansion regime. The
    # same holds for signed-unit scalars at any exponent parity, and whenever
    # every nonzero-order term annihilates every source monomial: only the
    # identity aggregate acts, so the power collapses to rescaling and the
    # unreachable expansion does not narrow the domain.
    scalar_action = _is_scalar_operator(operator)
    signed_unit = scalar_action and _is_signed_unit_scalar(operator)
    rescale_only = _rescale_only(polynomial, operator) or signed_unit
    if scalar_action or rescale_only:
        _require_nonexpanding_output(polynomial, expected)
    else:
        _require_expansion_operator(operator)
        _require_expansion_source(polynomial)
    _require_expected_output(expected)

    if rescale_only:
        expanded_terms = 0
        candidate_terms = len(polynomial.polynomial.terms)
    else:
        expanded_terms, candidate_terms, powered_orders = _expansion_support_candidates(
            polynomial,
            operator,
            iterations,
        )

    # Signed-unit scalar powers only flip signs: (±1)^k f = ±f keeps every
    # coefficient height unchanged, so their digit admission follows the
    # copied result's own heights rather than the common-denominator scaling
    # the generic estimate starts from.
    if signed_unit:
        coefficient_digits = _max_coefficient_digits(polynomial)
    else:
        coefficient_digits = _coefficient_digit_bound(
            polynomial,
            operator,
            iterations,
            candidate_terms,
            powered_orders if not rescale_only else None,
        )
    if coefficient_digits > MAX_APPLICATION_OUTPUT_COEFFICIENT_DIGITS:
        raise ValueError(
            "differential-operator output exceeds the coefficient-digit budget"
        )

    if rescale_only:
        # The result is c0^k * f: scaling costs one pass per source term and
        # the replay doubles it; no operator power or derivative expansion runs.
        work_units = 2 * 2 * len(polynomial.polynomial.terms)
    else:
        maximum_operator_order = max(
            (sum(term.orders) for term in operator.terms),
            default=0,
        )
        derivative_order = min(
            _total_degree(polynomial),
            iterations * maximum_operator_order,
        )
        term_count = len(operator.terms)
        maximum_axis_orders = tuple(
            max(term.orders[axis] for term in operator.terms)
            for axis in range(len(polynomial.variables))
        )
        power_work = _operator_power_work(term_count, iterations, maximum_axis_orders)
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
        rescale_only=rescale_only,
    )


__all__ = [
    "MAX_APPLICATION_OUTPUT_COEFFICIENT_DIGITS",
    "MAX_APPLICATION_OUTPUT_TERMS",
    "ApplicationEnvelope",
    "validate_application_envelope",
]
