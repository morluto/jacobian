"""Composable rational-function arithmetic and normalization bounds.

Consumers supply their work ledger, finite representation limits, and rejection
callback. These estimates own no operation-specific deadline or error policy.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from math import comb, gcd, isqrt, lcm
from typing import Literal, NoReturn, Protocol

from jacobian.canonical import format_canonical_integer
from jacobian.math.polynomials.values import RationalFunction, SparseRationalPolynomial

type BoundWorkCategory = Literal[
    "recognition",
    "source_conversion",
    "differentiation",
    "multiplication",
    "addition",
    "normalization",
]


@dataclass(frozen=True)
class RationalFunctionBoundLimits:
    """Owner-selected finite arithmetic and canonical representation limits."""

    raw_terms: int
    raw_digits: int
    result_exponent: int
    result_terms: int
    result_digits: int
    label: str
    reject: Callable[[str, str], NoReturn]


class BoundsLedger(Protocol):
    """An operation-owned work ledger used by the shared estimates."""

    limits: RationalFunctionBoundLimits

    def charge(self, category: BoundWorkCategory, amount: int) -> None: ...


@dataclass(frozen=True)
class PolynomialBound:
    """Bound ``rational_content * integral_polynomial`` exactly.

    Input bounds start with a primitive integral polynomial. Arithmetic may
    leave its integral factor nonprimitive, while ``coefficient_digits`` still
    bounds every coefficient and ``rational_content`` remains an exact common
    scale. Keeping that scale separate prevents unrelated coefficient
    denominators from multiplying during height admission.
    """

    terms: int
    degrees: tuple[int, ...]
    total_degree: int
    minimum_exponents: tuple[int, ...]
    coefficient_digits: int
    rational_content: Fraction

    @property
    def is_zero(self) -> bool:
        return self.terms == 0


@dataclass(frozen=True)
class FractionBound:
    numerator: PolynomialBound
    denominator: PolynomialBound

    @property
    def is_zero(self) -> bool:
        return self.numerator.is_zero


def _dense_term_bound(degrees: tuple[int, ...], *, cap: int | None = None) -> int:
    result = 1
    for degree in degrees:
        result *= degree + 1
        if cap is not None and result > cap:
            return cap + 1
    return result


def _total_degree_term_bound(bound: PolynomialBound) -> int:
    """Bound support using both coordinate degrees and total degree.

    Every factor has at most the source's total and coordinate degrees.
    Stars and bars counts all exponent vectors of total degree at most T
    in the active variables, including the constant monomial.
    """
    active = sum(degree > 0 for degree in bound.degrees)
    return min(
        _dense_term_bound(bound.degrees), comb(bound.total_degree + active, active)
    )


def _polynomial_admission_work_units(
    polynomial: SparseRationalPolynomial, variable_count: int
) -> int:
    """Price the scalar passes used to derive one exact polynomial bound.

    Each nonzero coefficient is converted, participates in denominator/content
    reduction, is scaled and made primitive, and is inspected for height. Each
    exponent is inspected for the maximum, minimum, and total degree. The
    final unit prices construction of the separated rational content. Zero is
    represented by one exact ``Fraction`` construction.
    """

    terms = len(polynomial.terms)
    if terms == 0:
        return 1
    return terms * (6 + 3 * variable_count) + 1


def _polynomial_backend_conversion_work_units(bound: PolynomialBound) -> int:
    """Price sparse coefficient conversion and dense SymPy ``Poly`` creation."""

    dense_coefficients = 1 if bound.is_zero else _dense_term_bound(bound.degrees)
    coefficient_digits = (
        bound.coefficient_digits
        + len(format_canonical_integer(abs(bound.rational_content.numerator)))
        + len(format_canonical_integer(bound.rational_content.denominator))
    )
    coefficient_chunks = max(1, (coefficient_digits + 31) // 32)
    return bound.terms * coefficient_chunks + dense_coefficients


def _polynomial_bound(polynomial: SparseRationalPolynomial) -> PolynomialBound:
    if not polynomial.terms:
        return PolynomialBound(
            terms=0,
            degrees=(),
            total_degree=0,
            minimum_exponents=(),
            coefficient_digits=1,
            rational_content=Fraction(0),
        )
    variable_count = len(polynomial.terms[0].exponents)
    coefficients = tuple(
        Fraction(*term.coefficient.as_integer_ratio()) for term in polynomial.terms
    )
    common_denominator = lcm(*(coefficient.denominator for coefficient in coefficients))
    integral_coefficients = tuple(
        coefficient.numerator * (common_denominator // coefficient.denominator)
        for coefficient in coefficients
    )
    content_numerator = gcd(
        *(abs(coefficient) for coefficient in integral_coefficients)
    )
    primitive_coefficients = tuple(
        coefficient // content_numerator for coefficient in integral_coefficients
    )
    return PolynomialBound(
        terms=len(polynomial.terms),
        degrees=tuple(
            max(term.exponents[axis] for term in polynomial.terms)
            for axis in range(variable_count)
        ),
        total_degree=max(sum(term.exponents) for term in polynomial.terms),
        minimum_exponents=tuple(
            min(term.exponents[axis] for term in polynomial.terms)
            for axis in range(variable_count)
        ),
        coefficient_digits=max(
            len(format_canonical_integer(abs(coefficient)))
            for coefficient in primitive_coefficients
        ),
        rational_content=Fraction(content_numerator, common_denominator),
    )


def _zero_polynomial(variable_count: int) -> PolynomialBound:
    return PolynomialBound(
        terms=0,
        degrees=(0,) * variable_count,
        total_degree=0,
        minimum_exponents=(0,) * variable_count,
        coefficient_digits=1,
        rational_content=Fraction(0),
    )


def _one_polynomial(variable_count: int) -> PolynomialBound:
    return PolynomialBound(
        terms=1,
        degrees=(0,) * variable_count,
        total_degree=0,
        minimum_exponents=(0,) * variable_count,
        coefficient_digits=1,
        rational_content=Fraction(1),
    )


def _zero_fraction(variable_count: int) -> FractionBound:
    return FractionBound(
        numerator=_zero_polynomial(variable_count),
        denominator=_one_polynomial(variable_count),
    )


def _fraction_bound(value: RationalFunction, ledger: BoundsLedger) -> FractionBound:
    variable_count = len(value.variables)
    ledger.charge(
        "source_conversion",
        _polynomial_admission_work_units(value.numerator, variable_count)
        + _polynomial_admission_work_units(value.denominator, variable_count),
    )
    numerator = (
        _zero_polynomial(variable_count)
        if not value.numerator.terms
        else _polynomial_bound(value.numerator)
    )
    result = FractionBound(
        numerator=numerator,
        denominator=_polynomial_bound(value.denominator),
    )
    ledger.charge(
        "source_conversion",
        _polynomial_backend_conversion_work_units(result.numerator)
        + _polynomial_backend_conversion_work_units(result.denominator),
    )
    return result


def _check_raw_polynomial(
    bound: PolynomialBound, limits: RationalFunctionBoundLimits
) -> None:
    if bound.terms > limits.raw_terms:
        limits.reject(
            "intermediate_support",
            f"{limits.label} polynomial expansion exceeds the "
            f"{limits.raw_terms}-term intermediate budget",
        )
    scaled_coefficient_digits = max(
        len(format_canonical_integer(abs(bound.rational_content.numerator)))
        + bound.coefficient_digits,
        len(format_canonical_integer(bound.rational_content.denominator)),
    )
    if scaled_coefficient_digits > limits.raw_digits:
        limits.reject(
            "intermediate_height",
            f"{limits.label} coefficient growth exceeds the "
            f"{limits.raw_digits}-digit intermediate budget",
        )


def _differentiate_polynomial(
    source: PolynomialBound,
    axis: int,
    ledger: BoundsLedger,
    *,
    active_terms: int,
    maximum_axis_exponent: int,
    minimum_exponents: tuple[int, ...],
) -> PolynomialBound:
    if source.is_zero or active_terms == 0:
        return _zero_polynomial(len(source.degrees))
    total_degree = source.total_degree - 1
    degrees = tuple(
        min(degree - int(i == axis), total_degree)
        for i, degree in enumerate(source.degrees)
    )
    result = PolynomialBound(
        terms=active_terms,
        degrees=degrees,
        total_degree=total_degree,
        minimum_exponents=minimum_exponents,
        coefficient_digits=(
            source.coefficient_digits
            + len(format_canonical_integer(maximum_axis_exponent))
        ),
        rational_content=source.rational_content,
    )
    _check_raw_polynomial(result, ledger.limits)
    return result


def _multiply_polynomials(
    left: PolynomialBound,
    right: PolynomialBound,
    ledger: BoundsLedger,
) -> PolynomialBound:
    if left.is_zero or right.is_zero:
        return _zero_polynomial(len(left.degrees))
    pair_count = left.terms * right.terms
    ledger.charge("multiplication", pair_count)
    degrees = tuple(
        left_degree + right_degree
        for left_degree, right_degree in zip(left.degrees, right.degrees, strict=True)
    )
    collision_count = min(left.terms, right.terms)
    product_digits = left.coefficient_digits + right.coefficient_digits
    if collision_count == 1:
        coefficient_digits = product_digits
    else:
        coefficient_digits = product_digits + len(
            format_canonical_integer(collision_count)
        )
    result = PolynomialBound(
        terms=min(pair_count, _dense_term_bound(degrees)),
        degrees=degrees,
        total_degree=left.total_degree + right.total_degree,
        minimum_exponents=tuple(
            left_exponent + right_exponent
            for left_exponent, right_exponent in zip(
                left.minimum_exponents,
                right.minimum_exponents,
                strict=True,
            )
        ),
        coefficient_digits=coefficient_digits,
        rational_content=left.rational_content * right.rational_content,
    )
    _check_raw_polynomial(result, ledger.limits)
    return result


def _add_polynomials(
    left: PolynomialBound,
    right: PolynomialBound,
    ledger: BoundsLedger,
) -> PolynomialBound:
    if left.is_zero:
        return right
    if right.is_zero:
        return left
    ledger.charge("addition", left.terms + right.terms)
    degrees = tuple(
        max(left_degree, right_degree)
        for left_degree, right_degree in zip(left.degrees, right.degrees, strict=True)
    )
    common_content = Fraction(
        gcd(
            abs(left.rational_content.numerator),
            abs(right.rational_content.numerator),
        ),
        lcm(
            left.rational_content.denominator,
            right.rational_content.denominator,
        ),
    )
    left_multiplier = left.rational_content / common_content
    right_multiplier = right.rational_content / common_content
    if left_multiplier.denominator != 1 or right_multiplier.denominator != 1:
        raise AssertionError(
            "rational polynomial content gcd did not divide both inputs"
        )

    def scaled_digits(coefficient_digits: int, multiplier: int) -> int:
        if abs(multiplier) <= 1:
            return coefficient_digits
        return coefficient_digits + len(format_canonical_integer(abs(multiplier)))

    result = PolynomialBound(
        terms=min(left.terms + right.terms, _dense_term_bound(degrees)),
        degrees=degrees,
        total_degree=max(left.total_degree, right.total_degree),
        minimum_exponents=tuple(
            min(left_exponent, right_exponent)
            for left_exponent, right_exponent in zip(
                left.minimum_exponents,
                right.minimum_exponents,
                strict=True,
            )
        ),
        coefficient_digits=max(
            scaled_digits(left.coefficient_digits, left_multiplier.numerator),
            scaled_digits(right.coefficient_digits, right_multiplier.numerator),
        )
        + 1,
        rational_content=common_content,
    )
    _check_raw_polynomial(result, ledger.limits)
    return result


def _differentiate_fraction(
    source_value: RationalFunction,
    source_bound: FractionBound,
    axis: int,
    ledger: BoundsLedger,
) -> FractionBound:
    ledger.charge(
        "differentiation",
        source_bound.numerator.terms + source_bound.denominator.terms,
    )
    numerator_active = tuple(
        term for term in source_value.numerator.terms if term.exponents[axis] > 0
    )
    denominator_active = tuple(
        term for term in source_value.denominator.terms if term.exponents[axis] > 0
    )
    if not numerator_active and not denominator_active:
        return _zero_fraction(len(source_bound.numerator.degrees))
    numerator_derivative = _differentiate_polynomial(
        source_bound.numerator,
        axis,
        ledger,
        active_terms=len(numerator_active),
        maximum_axis_exponent=max(
            (term.exponents[axis] for term in numerator_active), default=1
        ),
        minimum_exponents=tuple(
            min(
                term.exponents[coordinate] - int(coordinate == axis)
                for term in numerator_active
            )
            for coordinate in range(len(source_value.variables))
        )
        if numerator_active
        else (0,) * len(source_value.variables),
    )
    denominator_derivative = _differentiate_polynomial(
        source_bound.denominator,
        axis,
        ledger,
        active_terms=len(denominator_active),
        maximum_axis_exponent=max(
            (term.exponents[axis] for term in denominator_active), default=1
        ),
        minimum_exponents=tuple(
            min(
                term.exponents[coordinate] - int(coordinate == axis)
                for term in denominator_active
            )
            for coordinate in range(len(source_value.variables))
        )
        if denominator_active
        else (0,) * len(source_value.variables),
    )
    first = _multiply_polynomials(
        numerator_derivative, source_bound.denominator, ledger
    )
    second = _multiply_polynomials(
        source_bound.numerator, denominator_derivative, ledger
    )
    numerator = _add_polynomials(first, second, ledger)
    if numerator.is_zero:
        return _zero_fraction(len(source_bound.numerator.degrees))
    denominator = _multiply_polynomials(
        source_bound.denominator, source_bound.denominator, ledger
    )
    return FractionBound(numerator=numerator, denominator=denominator)


def _multiply_fractions(
    left: FractionBound,
    right: FractionBound,
    ledger: BoundsLedger,
) -> FractionBound:
    if left.is_zero or right.is_zero:
        return _zero_fraction(len(left.numerator.degrees))
    return FractionBound(
        numerator=_multiply_polynomials(left.numerator, right.numerator, ledger),
        denominator=_multiply_polynomials(left.denominator, right.denominator, ledger),
    )


def _add_fractions(
    left: FractionBound,
    right: FractionBound,
    ledger: BoundsLedger,
) -> FractionBound:
    if left.is_zero:
        return right
    if right.is_zero:
        return left
    left_scaled = _multiply_polynomials(left.numerator, right.denominator, ledger)
    right_scaled = _multiply_polynomials(right.numerator, left.denominator, ledger)
    numerator = _add_polynomials(left_scaled, right_scaled, ledger)
    if numerator.is_zero:
        return _zero_fraction(len(left.numerator.degrees))
    return FractionBound(
        numerator=numerator,
        denominator=_multiply_polynomials(left.denominator, right.denominator, ledger),
    )


def _factor_coefficient_digits(bound: PolynomialBound) -> int:
    """Bound coefficients of a primitive integral factor of ``bound``.

    For an integral factor F of P, multiplicativity of Mahler measure and
    M(P/F)>=1 give H(F)<=L(F)<=2**sum(deg_i(F))*M(F)
    <=2**sum(deg_i(P))*||P||_2. The alternative bound
    L(F)<=(a+1)**deg(F)*M(F), for a active variables, can be sharper.
    See Amoroso--Mignotte, Acta Arith.
    99 (2001), p. 1, https://doi.org/10.4064/aa99-1-1.
    The separated rational content is handled by the caller.
    """

    if bound.is_zero:
        return 1
    active = sum(degree > 0 for degree in bound.degrees)
    multiplier = min(1 << sum(bound.degrees), (active + 1) ** bound.total_degree)
    norm_multiplier = isqrt(bound.terms - 1) + 1
    return bound.coefficient_digits + len(
        format_canonical_integer(multiplier * norm_multiplier)
    )


def _remove_guaranteed_common_monomial(bound: FractionBound) -> FractionBound:
    """Apply exact valuation presolve before the canonical-result proof.

    The coordinatewise minimum exponent is a factor of every monomial. Its
    common part can therefore be canceled without expanding or factoring a
    polynomial, which materially widens monomial-denominator requests.
    """

    if bound.is_zero:
        return bound
    common = tuple(
        min(numerator, denominator)
        for numerator, denominator in zip(
            bound.numerator.minimum_exponents,
            bound.denominator.minimum_exponents,
            strict=True,
        )
    )
    if not any(common):
        return bound

    def divide(polynomial: PolynomialBound) -> PolynomialBound:
        return PolynomialBound(
            terms=polynomial.terms,
            degrees=tuple(
                degree - exponent
                for degree, exponent in zip(polynomial.degrees, common, strict=True)
            ),
            total_degree=polynomial.total_degree - sum(common),
            minimum_exponents=tuple(
                minimum - exponent
                for minimum, exponent in zip(
                    polynomial.minimum_exponents, common, strict=True
                )
            ),
            coefficient_digits=polynomial.coefficient_digits,
            rational_content=polynomial.rational_content,
        )

    return FractionBound(
        numerator=divide(bound.numerator), denominator=divide(bound.denominator)
    )


def _guaranteed_linear_power_gcd(
    denominator: SparseRationalPolynomial,
) -> tuple[int, int]:
    """Return ``(axis, deg(gcd(q, q')))`` for a univariate binomial power.

    In characteristic zero, ``q = (alpha x_i^g + beta)^n`` with ``n >= 2`` and
    ``beta != 0`` has ``gcd(q, q')`` of degree ``(n-1) g``. The binomial
    coefficient recurrence is a source-intrinsic identity, so this lower bound
    does not replay differentiation or polynomial GCD.
    """

    if not denominator.terms:
        return -1, 0
    axes = len(denominator.terms[0].exponents)
    used = [
        axis
        for axis in range(axes)
        if any(term.exponents[axis] for term in denominator.terms)
    ]
    if len(used) != 1:
        return -1, 0
    axis = used[0]
    by_degree: dict[int, Fraction] = {}
    for term in denominator.terms:
        if any(term.exponents[other] for other in range(axes) if other != axis):
            return -1, 0
        degree = term.exponents[axis]
        if degree in by_degree:
            return -1, 0
        by_degree[degree] = term.coefficient.as_fraction()
    exponents = sorted(by_degree)
    if 0 not in by_degree:
        return -1, 0
    step = 0
    for exponent in by_degree:
        if exponent:
            step = exponent if step == 0 else gcd(step, exponent)
    if step < 1:
        return -1, 0
    power = max(by_degree) // step
    if power < 2 or exponents != list(range(0, power * step + 1, step)):
        return -1, 0
    constant = by_degree[0]
    if constant == 0:
        return -1, 0
    scale = by_degree[step] / (constant * power)
    for index in range(1, power + 1):
        expected = (
            by_degree[(index - 1) * step] * Fraction(power - index + 1, index) * scale
        )
        if by_degree[index * step] != expected:
            return -1, 0
    return axis, (power - 1) * step


def _remove_guaranteed_linear_power_factor(
    bound: FractionBound, source: RationalFunction
) -> FractionBound:
    """Cancel the forced ``gcd(q, q')`` factor before canonical-result caps."""

    axis, extra = _guaranteed_linear_power_gcd(source.denominator)
    if extra <= 0 or bound.is_zero:
        return bound

    def reduce(polynomial: PolynomialBound) -> PolynomialBound:
        if axis >= len(polynomial.degrees):
            return polynomial
        drop = min(extra, polynomial.degrees[axis], polynomial.total_degree)
        if drop <= 0:
            return polynomial
        degrees = tuple(
            max(0, degree - drop) if index == axis else degree
            for index, degree in enumerate(polynomial.degrees)
        )
        return PolynomialBound(
            terms=polynomial.terms,
            degrees=degrees,
            total_degree=max(0, polynomial.total_degree - drop),
            minimum_exponents=polynomial.minimum_exponents,
            coefficient_digits=polynomial.coefficient_digits,
            rational_content=polynomial.rational_content,
        )

    return FractionBound(
        numerator=reduce(bound.numerator), denominator=reduce(bound.denominator)
    )


def _canonical_coefficient_digits(bound: FractionBound) -> int:
    """Bound monic reduction without counting integral content twice.

    Write the raw pair as cN*N, cD*D with integral N,D. A primitive integral
    common gcd G leaves integral factors A=N/G and B=D/G; their contents are
    already included in the factor-height bounds. Canonical coefficients are
    (cN/cD)*A_i/LC(B) and B_i/LC(B), so their rational components need only
    the corresponding factor height and the separated content ratio.
    """
    if bound.is_zero:
        return 1
    content_ratio = (
        bound.numerator.rational_content / bound.denominator.rational_content
    )
    denominator_is_unit = all(degree == 0 for degree in bound.denominator.degrees)
    numerator_factor_digits = (
        bound.numerator.coefficient_digits
        if denominator_is_unit
        else _factor_coefficient_digits(bound.numerator)
    )
    denominator_factor_digits = (
        bound.denominator.coefficient_digits
        if denominator_is_unit
        else _factor_coefficient_digits(bound.denominator)
    )
    return max(
        len(format_canonical_integer(abs(content_ratio.numerator)))
        + numerator_factor_digits,
        len(format_canonical_integer(content_ratio.denominator))
        + denominator_factor_digits,
        denominator_factor_digits,
    )


def _validate_canonical_result_bound(
    bound: FractionBound,
    ledger: BoundsLedger,
    *,
    work_bound: FractionBound | None = None,
) -> int:
    limits = ledger.limits
    charged = bound if work_bound is None else work_bound
    if bound.is_zero:
        ledger.charge("normalization", 1)
        return 1
    for label, polynomial in (
        ("numerator", bound.numerator),
        ("denominator", bound.denominator),
    ):
        if any(degree > limits.result_exponent for degree in polynomial.degrees):
            limits.reject(
                "result_exponent",
                f"{limits.label} {label} can exceed the canonical "
                f"exponent bound {limits.result_exponent}",
            )
        dense_terms = _total_degree_term_bound(polynomial)
        # When the denominator is the unit polynomial, there can be no
        # cancellation-induced support expansion, so the tracked sparse
        # term count is the accurate support bound.
        denominator_is_unit = all(degree == 0 for degree in bound.denominator.degrees)
        support_terms = polynomial.terms if denominator_is_unit else dense_terms
        if support_terms > limits.result_terms:
            limits.reject(
                "result_support",
                f"{limits.label} {label} can exceed the canonical "
                f"{limits.result_terms}-term bound",
            )
    coefficient_digits = _canonical_coefficient_digits(bound)
    if coefficient_digits > limits.result_digits:
        limits.reject(
            "result_height",
            f"{limits.label} normalization can exceed the canonical "
            f"{limits.result_digits}-digit coefficient bound",
        )
    charged_denominator_is_unit = all(
        degree == 0 for degree in charged.denominator.degrees
    )
    # Normalization still uses the recursively dense backend. A sparse
    # canonical support bound does not justify reducing this work charge.
    numerator_dense = (
        charged.numerator.terms
        if charged_denominator_is_unit
        else _dense_term_bound(charged.numerator.degrees)
    )
    denominator_dense = (
        1
        if charged_denominator_is_unit
        else _dense_term_bound(charged.denominator.degrees)
    )
    work_digits = _canonical_coefficient_digits(charged)
    normalization_degree = max(numerator_dense + denominator_dense - 2, 0)
    ledger.charge(
        "normalization",
        (numerator_dense + denominator_dense)
        * (normalization_degree + 1)
        * work_digits,
    )
    return coefficient_digits


def _recognition_work_units(bound: FractionBound) -> int:
    """Charge the dense exact coefficient work exposed to SymPy's GCD.

    SymPy 1.14 represents a multivariate ``Poly`` as a recursively dense DMP.
    The product of per-axis degree ranges therefore bounds the coefficients
    materialized by conversion.  Total degree steps and 32-digit coefficient
    chunks conservatively price the subsequent exact GCD reductions.  A
    killable worker still owns wall time and memory because SymPy exposes no
    cooperative cancellation hook inside ``Poly.gcd``.
    """

    numerator_dense = _dense_term_bound(bound.numerator.degrees)
    denominator_dense = _dense_term_bound(bound.denominator.degrees)
    degree_steps = (
        sum(
            max(numerator, denominator)
            for numerator, denominator in zip(
                bound.numerator.degrees,
                bound.denominator.degrees,
                strict=True,
            )
        )
        + 1
    )
    coefficient_digits = max(
        bound.numerator.coefficient_digits
        + len(format_canonical_integer(abs(bound.numerator.rational_content.numerator)))
        + len(format_canonical_integer(bound.numerator.rational_content.denominator)),
        bound.denominator.coefficient_digits
        + len(
            format_canonical_integer(abs(bound.denominator.rational_content.numerator))
        )
        + len(format_canonical_integer(bound.denominator.rational_content.denominator)),
    )
    coefficient_chunks = max(1, (coefficient_digits + 31) // 32)
    return (numerator_dense + denominator_dense) * degree_steps * coefficient_chunks
