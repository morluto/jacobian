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
    work_cap: int,
) -> bool:
    """Recognize requests whose only acting aggregate is the identity order.

    Every nonzero-order operator term must annihilate every source monomial,
    so the powered operator's action collapses to rescaling the source by
    the zero-order coefficient raised to the iteration count. The pairwise
    annihilation scan is charged against the request's declared deterministic
    work budget; exceeding that budget conservatively reports the expanding
    regime instead of trusting a partial scan.
    """

    if not operator.terms or not polynomial.polynomial.terms:
        return False
    work = 0
    for term in operator.terms:
        if not any(term.orders):
            continue
        for monomial in polynomial.polynomial.terms:
            work += 1
            if work > work_cap:
                return False
            if all(
                order <= exponent
                for order, exponent in zip(term.orders, monomial.exponents, strict=True)
            ):
                return False
    return True


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


def _merged_unit_power_weights(
    operator: ConstantCoefficientDifferentialOperator,
    iterations: int,
    cap: int,
) -> dict[tuple[int, ...], int] | None:
    """Enumerate the powered operator's shifts with exact merged weights.

    The caller has established that every operator coefficient is the integer
    unit ``±1``, so each composition ``m`` of ``iterations`` among the operator
    terms contributes the signed multinomial coefficient
    ``(k!/m_1!...m_t!)·Π(±1)^m_j`` to the shift ``Σ m_j·orders_j``, and no
    other magnitude enters. The expanded-support budget bounds the composition
    family; exceeding ``cap`` returns ``None`` so admission falls back to the
    conservative generic estimate instead of trusting a partial enumeration.
    """

    weights: dict[tuple[int, ...], int] = {}
    visited = 0
    if iterations == 1:
        # A single application selects exactly one term per composition: each
        # term contributes its own shift with signed unit weight.  Building
        # this family directly avoids the skip-chain DFS whose frame cost is
        # quadratic in the term count and crashed workers on wide operators.
        for term in operator.terms:
            sign = 1 if term.coefficient.as_fraction() == 1 else -1
            shift = tuple(term.orders)
            weights[shift] = weights.get(shift, 0) + sign
        return weights
    work = 0
    frames: list[tuple[int, int, tuple[int, ...], int, int]] = [
        (0, iterations, (0,) * len(operator.variables), 0, 1)
    ]
    while frames:
        index, remaining, shift, chosen, weight = frames.pop()
        work += 1
        if work > ENUMERATION_WORK_CAP:
            return None
        if index == len(operator.terms):
            if remaining:
                continue
            visited += 1
            if visited > cap:
                return None
            weights[shift] = weights.get(shift, 0) + weight
            continue
        orders = operator.terms[index].orders
        continuations: tuple[tuple[int, int], ...]
        if index + 1 == len(operator.terms):
            # The final term must consume every remaining iteration, so the
            # composition family stays exactly at the counted support size.
            continuations = ((remaining, math.comb(chosen + remaining, remaining)),)
        else:
            continuations = tuple(
                (taken, math.comb(chosen + taken, taken))
                for taken in range(remaining + 1)
            )
        for taken, factor in continuations:
            frames.append(
                (
                    index + 1,
                    remaining - taken,
                    tuple(
                        current + taken * order
                        for current, order in zip(shift, orders, strict=True)
                    ),
                    chosen + taken,
                    weight * factor,
                )
            )
    return weights


def _derivative_regime_preserves_coefficient_heights(
    polynomial: RationalPolynomial,
    operator: ConstantCoefficientDifferentialOperator,
    iterations: int,
) -> bool:
    """Recognize derivative powers that provably preserve coefficient height.

    With unit integer operator coefficients, each merged shift weight is an
    exact integer, so every surviving output coefficient is an input
    coefficient times that integer times a falling-factorial product. When
    every such product along every admitted shift annihilates its monomial or
    equals one, no merged weight exceeds one in absolute value, and no output
    monomial receives more than one surviving contribution, each output
    coefficient is zero or the unchanged source coefficient: no numerator
    grows and no denominator appears. Differentiation then cannot raise the
    height, so admission follows the input's own heights rather than any
    residual estimate slack. The composition family itself bounds the exact
    enumeration, and overflowing it conservatively reports a growing regime.
    """

    if any(term.coefficient.as_fraction() not in (1, -1) for term in operator.terms):
        return False
    compositions = _bounded_multiset_count(
        len(operator.terms),
        iterations,
        MAX_APPLICATION_OUTPUT_TERMS,
    )
    if compositions > MAX_APPLICATION_OUTPUT_TERMS:
        return False
    weights = _merged_unit_power_weights(operator, iterations, compositions)
    if weights is None:
        return False
    seen_outputs: set[tuple[int, ...]] = set()
    for term in polynomial.polynomial.terms:
        for shift, weight in weights.items():
            multiplier = abs(weight)
            for order, exponent in zip(shift, term.exponents, strict=True):
                if not order:
                    continue
                if exponent < order:
                    multiplier = 0
                    break
                if order != 1 or exponent != 1:
                    return False
            if multiplier > 1:
                return False
            if multiplier:
                # Two source monomials merging onto one output coefficient can
                # double the height even when each path alone preserves it.
                output = tuple(
                    exponent - order
                    for exponent, order in zip(term.exponents, shift, strict=True)
                )
                if output in seen_outputs:
                    return False
                seen_outputs.add(output)
    return True


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
_HEIGHT_CAP_BITS = 64 * MAX_APPLICATION_OUTPUT_COEFFICIENT_DIGITS
# A single catalog invocation validates the request, runs the compute
# preflight, validates the inherited result, and runs the replay preflight;
# per-pass admission scans share one quarter of the deterministic budget.
_ADMISSION_SCAN_PASSES = 4
_RETAINED_WEIGHT_BITS = MAX_APPLICATION_RESULT_BYTES * 8


def _distinct_powered_orders(
    operator: ConstantCoefficientDifferentialOperator,
    iterations: int,
    limit: int,
    work_cap: int,
) -> tuple[tuple[int, ...], ...] | None:
    """Enumerate the distinct aggregate multi-indices of a powered operator.

    Returns ``None`` when enumeration exceeds ``limit`` distinct sums or
    ``work_cap`` steps; callers then fall back to the analytic support bound.
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
                if len(expanded) > limit or work > work_cap:
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


def _common_denominator_height(
    values: Iterable[Fraction],
    *,
    maximum_digits: int | None = None,
) -> tuple[int, int] | None:
    """Shared denominator and largest scaled numerator of ``values``.

    Every later shared denominator is a multiple of the running one, so once
    its decimal length exceeds ``maximum_digits`` no completion can stay
    inside that digit budget; construction stops there and returns ``None``
    instead of materializing an arbitrarily large integer.
    """

    fractions = tuple(values)
    if not fractions:
        return 1, 0
    denominator = 1
    for value in fractions:
        denominator = math.lcm(denominator, value.denominator)
        if (
            maximum_digits is not None
            and _decimal_digits_from_bits(denominator.bit_length()) > maximum_digits
        ):
            return None
    maximum_scaled_numerator = max(
        abs(value.numerator) * (denominator // value.denominator) for value in fractions
    )
    return denominator, maximum_scaled_numerator


def _decimal_digits_from_bits(bits: int) -> int:
    # 30103 / 100000 is a strict upper rational approximation to log10(2).
    return max(1, (bits * 30_103 + 99_999) // 100_000)


def _height_decimal_digits(bits: int, value: int) -> int:
    """Exact decimal length of ``value`` without string conversion.

    The bit-based estimate is a strict upper bound, so it answers directly
    whenever it sits below the output budget; near or above the budget the
    exact length is resolved by comparing against powers of ten.
    """

    estimate = _decimal_digits_from_bits(bits)
    if estimate < MAX_APPLICATION_OUTPUT_COEFFICIENT_DIGITS:
        return estimate
    magnitude = abs(value)
    if magnitude == 0:
        return 1
    digits = estimate
    threshold = 10 ** (digits - 1)
    while magnitude < threshold:
        digits -= 1
        threshold //= 10
    return digits


def _max_coefficient_digits(polynomial: RationalPolynomial) -> int:
    return max(
        (
            max(len(term.coefficient.num.lstrip("-")), len(term.coefficient.den))
            for term in polynomial.polynomial.terms
        ),
        default=1,
    )


def _shift_weights(
    operator: ConstantCoefficientDifferentialOperator,
    iterations: int,
    support_cap: int,
) -> dict[tuple[int, ...], Fraction] | None:
    """Enumerate exact rational weights per distinct shift of the power.

    Weights carry operator coefficients and multinomial multiplicities
    exactly. Returns ``None`` when enumeration exceeds ``support_cap``
    distinct shifts or the work cap.
    """

    terms = tuple(
        (term.orders, term.coefficient.as_fraction()) for term in operator.terms
    )
    if not terms:
        return None
    zero_shift = tuple(0 for _ in range(len(terms[0][0])))
    weights = {zero_shift: Fraction(1)}
    work = 1
    retained_bits = 1
    for _ in range(iterations):
        merged: dict[tuple[int, ...], Fraction] = {}
        exhausted = False
        for shift, weight in weights.items():
            for orders, coefficient in terms:
                shifted = tuple(
                    current + order
                    for current, order in zip(shift, orders, strict=True)
                )
                contribution = weight * coefficient
                entry = merged.get(shifted)
                merged[shifted] = (
                    contribution if entry is None else entry + contribution
                )
                work += 1
                if (
                    len(merged) > support_cap
                    or work > ENUMERATION_WORK_CAP
                    or contribution.numerator.bit_length() > _HEIGHT_CAP_BITS
                    or contribution.denominator.bit_length() > _HEIGHT_CAP_BITS
                    or merged[shifted].numerator.bit_length() > _HEIGHT_CAP_BITS
                    or merged[shifted].denominator.bit_length() > _HEIGHT_CAP_BITS
                ):
                    exhausted = True
                    break
            if exhausted:
                break
        if exhausted:
            return None
        retained_bits += sum(
            shifted_weight.numerator.bit_length()
            + shifted_weight.denominator.bit_length()
            for shifted_weight in merged.values()
        )
        if retained_bits > _RETAINED_WEIGHT_BITS:
            return None
        weights = merged
    return weights


def _falling_factorial_product(
    exponents: tuple[int, ...],
    orders: tuple[int, ...],
    cap_bits: int,
    work: list[int],
) -> int | None:
    """Exact falling-factorial product for one shift on one monomial.

    Returns ``None`` when the product itself crosses ``cap_bits`` or the
    shared falling-factorial step budget is exhausted.
    """

    product = 1
    for exponent, order in zip(exponents, orders, strict=True):
        if not order:
            continue
        for factor in range(exponent - order + 1, exponent + 1):
            product *= factor
            work[0] += 1
            if product.bit_length() > cap_bits or work[0] > ENUMERATION_WORK_CAP:
                return None
    return product


def _per_exponent_height_bits(
    polynomial: RationalPolynomial,
    operator: ConstantCoefficientDifferentialOperator,
    iterations: int,
    support_cap: int,
) -> tuple[int, int] | None:
    """Bound output coefficient heights per colliding exponent class.

    The powered operator is enumerated once with exact rational weights per
    shift (operator coefficients and multinomial multiplicities included), so
    no global operator LCM enters: contributions merge only when
    differentiated source monomials land on the same output exponent, and
    each class combines denominators through its own least common multiple.
    Falling factorials are measured exactly per contribution instead of
    substituting exponent^order, which overestimates boundary cases such as
    (8500)_8500 = 8500! by thousands of digits. Returns ``None`` when the
    accounting or its falling-factorial work would exceed its caps, in which
    case the caller falls back to the coarser global bound.
    """

    cap = _HEIGHT_CAP_BITS
    falling_work = [0]
    weights = _shift_weights(operator, iterations, support_cap)
    if weights is None:
        return None
    classes: dict[tuple[int, ...], tuple[int, int]] = {}
    for shift, weight in weights.items():
        if (
            abs(weight.numerator).bit_length() > cap
            or weight.denominator.bit_length() > cap
        ):
            return None
        for source_term in polynomial.polynomial.terms:
            if any(
                order > exponent
                for order, exponent in zip(shift, source_term.exponents, strict=True)
            ):
                continue
            target = tuple(
                exponent - order
                for exponent, order in zip(source_term.exponents, shift, strict=True)
            )
            falling_product = _falling_factorial_product(
                source_term.exponents, shift, cap, falling_work
            )
            if falling_product is None:
                # True growth provably crosses the height envelope; the coarse
                # global estimate rejects this request soundly.
                return None
            # The exact reduced contribution folds the falling factorial in
            # before normalization, so cross-canceling factors - an N weight
            # against a 1/N coefficient, or 10000! against a /2000!
            # denominator - never inflate the accounted heights. Numerators
            # stay signed: contributions to one target exponent are summed
            # exactly, so additive cancellation such as N + (-N) is retained
            # instead of bounding the class by the sum of absolute values.
            contribution = (
                weight * source_term.coefficient.as_fraction() * falling_product
            )
            class_denominator = contribution.denominator
            scaled_numerator = contribution.numerator
            entry = classes.get(target)
            if entry is None:
                if class_denominator.bit_length() > cap:
                    return None
                classes[target] = (class_denominator, scaled_numerator)
                continue
            lcm, numerator_sum = entry
            new_lcm = math.lcm(lcm, class_denominator)
            total = numerator_sum * (new_lcm // lcm) + scaled_numerator * (
                new_lcm // class_denominator
            )
            if new_lcm.bit_length() > cap or total.bit_length() > cap:
                return None
            classes[target] = (new_lcm, total)
    worst_digits = 0
    for lcm, numerator_sum in classes.values():
        worst_digits = max(
            worst_digits,
            _height_decimal_digits(lcm.bit_length(), lcm),
            _height_decimal_digits(numerator_sum.bit_length(), numerator_sum),
        )
    return worst_digits, 0


def _coefficient_digit_bound(
    polynomial: RationalPolynomial,
    operator: ConstantCoefficientDifferentialOperator,
    iterations: int,
    candidate_terms: int,
    support_cap: int,
) -> int:
    if iterations == 0:
        return _max_coefficient_digits(polynomial)

    # The refined per-exponent accounting never constructs a shared height
    # across nonacting terms, so it runs first and answers without any
    # global operator arithmetic.
    per_exponent = _per_exponent_height_bits(
        polynomial,
        operator,
        iterations,
        support_cap,
    )
    if per_exponent is not None:
        return max(per_exponent[0], per_exponent[1])

    operator_height = _common_denominator_height(
        (term.coefficient.as_fraction() for term in operator.terms),
        maximum_digits=MAX_APPLICATION_OUTPUT_COEFFICIENT_DIGITS,
    )
    source_height = _common_denominator_height(
        (term.coefficient.as_fraction() for term in polynomial.polynomial.terms),
        maximum_digits=MAX_APPLICATION_OUTPUT_COEFFICIENT_DIGITS,
    )
    if operator_height is None or source_height is None:
        # A shared denominator whose own decimal length already forces more
        # than budget digits onto every coefficient bound rejects the request
        # either way, so the coarse bound returns one digit past the budget
        # and no intermediate larger than the gate ever materializes.
        return MAX_APPLICATION_OUTPUT_COEFFICIENT_DIGITS + 1
    operator_denominator, operator_numerator = operator_height
    source_denominator, source_numerator = source_height
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
    # iterations sequences. A one-bit numerator bound means every operator
    # coefficient is the unit +-1 and adds no growth, matching the tight
    # multiplier bound's treatment of unit factors.
    operator_growth_bits = _multiplier_bit_bound(operator_numerator, iterations)
    shared_numerator_bits = (
        _multiplier_bit_bound(candidate_terms, 1)
        + _multiplier_bit_bound(len(operator.terms), iterations)
        + operator_growth_bits
        + falling_factor_bits
    )
    shared_denominator_bits = _multiplier_bit_bound(operator_denominator, iterations)

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
    # The correlated-sum enumeration may cost about as much as the sparse
    # power whose support it predicts, so its cutoff follows the admitted
    # power-work budget: a power that already exhausts the deterministic work
    # budget rejects regardless of its exact support and skips straight to
    # the analytic bound.
    power_work = _operator_power_work(term_count, iterations, maximum_axis_orders)
    powered_orders = (
        _distinct_powered_orders(
            operator,
            iterations,
            MAX_APPLICATION_OUTPUT_TERMS,
            power_work,
        )
        if power_work <= MAX_APPLICATION_WORK_UNITS
        else None
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
    # The coarse deterministic-work gate uses conservative candidate bounds so
    # requests whose expansion cannot fit the work budget reject before any
    # per-pair candidate scanning runs.
    source_terms = len(polynomial.polynomial.terms)
    coarse_candidates = source_terms * expanded_terms
    maximum_operator_order = max(sum(term.orders) for term in operator.terms)
    derivative_order = min(
        _total_degree(polynomial),
        iterations * maximum_operator_order,
    )
    derivative_work = source_terms * expanded_terms * (1 + derivative_order)
    conversion_work = 2 * (source_terms + term_count + coarse_candidates)
    if (
        2 * (power_work + derivative_work) + conversion_work
        > MAX_APPLICATION_WORK_UNITS
    ):
        raise ValueError(
            "differential-operator application exceeds the deterministic work budget"
        )
    # A powered aggregate acts on a source monomial only when every axis
    # order stays within that monomial's exponents, so the candidate support
    # counts the distinct output exponents across acting aggregates per source
    # monomial instead of treating annihilating powered terms as surviving
    # output paths. Distinct targets - not acting pairs - bound the serialized
    # result: colliding contributions such as an identity shift and a
    # derivative shift landing on one degree merge into a single output term.
    if powered_orders is not None:
        targets: set[tuple[int, ...]] = set()
        for source_term in polynomial.polynomial.terms:
            for orders in powered_orders:
                if not all(
                    order <= exponent
                    for order, exponent in zip(
                        orders, source_term.exponents, strict=True
                    )
                ):
                    continue
                targets.add(
                    tuple(
                        exponent - order
                        for exponent, order in zip(
                            source_term.exponents, orders, strict=True
                        )
                    )
                )
                if len(targets) > MAX_APPLICATION_OUTPUT_TERMS:
                    raise ValueError(
                        "differential-operator output exceeds the candidate-term budget"
                    )
        return expanded_terms, len(targets), powered_orders
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


def _rescaled_height_digits(
    polynomial: RationalPolynomial,
    operator: ConstantCoefficientDifferentialOperator,
    iterations: int,
) -> int | None:
    """Digits of the largest reduced coefficient of ``c0**k * f``.

    Binary exponentiation normalizes at every step, so cross-canceling
    factors - a scalar weight against a source denominator - reduce
    immediately. Scaling preserves the source's nonzero pattern, so no two
    terms merge and the per-term maximum is exact. Returns ``None`` when an
    intermediate provably exceeds the height cap; callers then fall back to
    the sound additive bound.
    """

    cap_bits = _HEIGHT_CAP_BITS
    zero_coefficient = next(
        (
            term.coefficient.as_fraction()
            for term in operator.terms
            if not any(term.orders)
        ),
        Fraction(0),
    )
    worst_digits = 0
    for term in polynomial.polynomial.terms:
        value = term.coefficient.as_fraction()
        result = Fraction(1)
        base = abs(zero_coefficient)
        exponent = iterations
        while exponent:
            if exponent & 1:
                result *= base
                if (
                    result.numerator.bit_length() > cap_bits
                    or result.denominator.bit_length() > cap_bits
                ):
                    return None
            exponent >>= 1
            if exponent:
                base *= base
                if (
                    base.numerator.bit_length() > cap_bits
                    or base.denominator.bit_length() > cap_bits
                ):
                    return None
        value *= result
        worst_digits = max(
            worst_digits,
            _height_decimal_digits(value.numerator.bit_length(), abs(value.numerator)),
            _height_decimal_digits(value.denominator.bit_length(), value.denominator),
        )
    return worst_digits


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
    rescale_only = _rescale_only(
        polynomial,
        operator,
        MAX_APPLICATION_WORK_UNITS // _ADMISSION_SCAN_PASSES,
    )
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
        expanded_terms, candidate_terms, _ = _expansion_support_candidates(
            polynomial,
            operator,
            iterations,
        )

    # Signed-unit scalar powers only flip signs, and every rescale-only regime
    # applies exactly one zero-order coefficient to the k-th power per source
    # term. The exact reduced product bounds each output height - excluding
    # annihilating coefficients and capturing cancellations with source
    # denominators - and falls back to the sound additive bound when an
    # intermediate provably overflows the height cap.
    if rescale_only:
        exact_digits = _rescaled_height_digits(polynomial, operator, iterations)
        if exact_digits is not None:
            coefficient_digits = exact_digits
        else:
            zero_coefficient = next(
                (
                    term.coefficient.as_fraction()
                    for term in operator.terms
                    if not any(term.orders)
                ),
                Fraction(0),
            )
            growth_bits = _multiplier_bit_bound(
                abs(zero_coefficient.numerator), iterations
            ) + _multiplier_bit_bound(zero_coefficient.denominator, iterations)
            growth_digits = _decimal_digits_from_bits(growth_bits) if growth_bits else 0
            coefficient_digits = _max_coefficient_digits(polynomial) + growth_digits
    elif not scalar_action and _derivative_regime_preserves_coefficient_heights(
        polynomial,
        operator,
        iterations,
    ):
        # Unit derivative powers that only keep or annihilate each source
        # coefficient preserve every height exactly, so a boundary request
        # such as f = 10^32767·x, D = ∂x, k = 1 is admitted at the input's
        # own height instead of any residual estimate slack.
        coefficient_digits = _max_coefficient_digits(polynomial)
    else:
        coefficient_digits = _coefficient_digit_bound(
            polynomial,
            operator,
            iterations,
            candidate_terms,
            MAX_APPLICATION_OUTPUT_TERMS if not rescale_only else 0,
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
