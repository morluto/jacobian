"""Exact bounded unit-circle supremum norms of rational polynomials.

The kernel reduces ``max_{|z|=1} |P(z)|^2`` to a real extremum of an exact
rational function on the parametrized circle.  Writing

    z = (1 + i t) / (1 - i t),      t in R,

the numerator ``W(t) = sum_k c_k (1 + i t)**k (1 - i t)**(n-k)`` is a
Gaussian-rational polynomial of degree ``n``, and

    |P(z)|^2 = Q(t) / (1 + t**2)**n,   Q(t) = Re(W)**2 + Im(W)**2.

The finite critical points are the real roots of the exact derivative
numerator ``D(t) = Q'(t)(1 + t**2) - 2 n t Q(t)``; the omitted circle point
``z = -1`` is the endpoint ``t = infinity``.  Real-root isolation and exact
value comparison are provided by the fixed SymPy backend.  Critical values
are identified with the real roots of the exact resultant of ``D`` (with any
reducible ``1 + t**2`` factor removed) and ``y (1 + t**2)**n - Q``; equal
values therefore share a resultant root and are never confused by floating
point.
"""

from __future__ import annotations

import time
from bisect import bisect_left
from fractions import Fraction
from itertools import pairwise
from math import gcd, isqrt, lcm
from typing import Any

import sympy

from jacobian._exact import CanonicalRational
from jacobian._execution import (
    BackendFailureReason,
    OperationBackendError,
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
    request_execution,
)
from jacobian.canonical import decimal_digit_width
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math._root_isolation import strict_root_count
from jacobian.math.number_theory.algebraic_numbers.real import (
    MAX_REAL_ALGEBRAIC_COEFFICIENT_DIGITS,
    MAX_REAL_ALGEBRAIC_DEGREE,
    RationalIsolatingInterval,
    RealAlgebraicValue,
)
from jacobian.math.polynomials.unit_circle._sup_norm_bounds import sup_norm_bounds
from jacobian.math.polynomials.unit_circle._sup_norm_models import (
    MAX_SUP_NORM_CERTIFICATE_SCALAR_DIGITS,
    MAX_SUP_NORM_COMPARISONS,
    MAX_SUP_NORM_CRITICAL_ROOTS,
    MAX_SUP_NORM_DEGREE,
    MAX_SUP_NORM_DERIVATIVE_COMPONENT_DIGITS,
    MAX_SUP_NORM_FACTOR_WORK,
    MAX_SUP_NORM_INPUT_COMPONENT_DIGITS,
    MAX_SUP_NORM_ISOLATION_WORK,
    MAX_SUP_NORM_PARAMETER_COMPONENT_DIGITS,
    MAX_SUP_NORM_PARAMETER_DECIMAL_PLACES,
    MAX_SUP_NORM_POSTPROCESSING_WORK,
    MAX_SUP_NORM_RESULTANT_COMPONENT_DIGITS,
    MAX_SUP_NORM_RESULTANT_WORK,
    MAX_SUP_NORM_TRANSFORM_COMPONENT_DIGITS,
    MAX_SUP_NORM_TRANSFORM_WORK,
    MAX_SUP_NORM_VALUE_ANALYSIS_WORK,
    GaussianRationalPolynomial,
    UnitCircleCriticalPoint,
    UnitCircleSupNormSquaredResult,
)

__all__ = [
    "unit_circle_sup_norm_squared",
    "verify_unit_circle_sup_norm_squared",
]

# Rational endpoints of the reported enclosures are refined until the
# parameter width is below this target. Reported nonsingleton value/image
# enclosures are compacted outward after exact ordering; root parameters and
# exact rational singletons retain their own source-derived growth bounds.
_REFINEMENT_TARGETS = (Fraction(1, 10**40), Fraction(1, 10**70), Fraction(1, 10**100))
_REPORTING_SCALE = 10**100


def _resource(reason: str, message: str) -> OperationResourceAdmissionError:
    return OperationResourceAdmissionError(
        location=("polynomial",),
        code=f"polynomial.unit_circle.sup_norm.{reason}",
        message=message,
    )


def _invalid_backend_output(message: str) -> OperationBackendError:
    return OperationBackendError(BackendFailureReason.INVALID_OUTPUT)


def _digits(value: Fraction) -> int:
    return max(
        decimal_digit_width(value.numerator), decimal_digit_width(value.denominator)
    )


def _interval(lower: Fraction, upper: Fraction) -> RationalIsolatingInterval:
    return RationalIsolatingInterval(
        lower=CanonicalRational(num=lower.numerator, den=lower.denominator),
        upper=CanonicalRational(num=upper.numerator, den=upper.denominator),
        interval_type="SINGLETON" if lower == upper else "OPEN",
    )


def _rational_interval(value: Fraction) -> RationalIsolatingInterval:
    return _interval(value, value)


def _reported_interval(lower: Fraction, upper: Fraction) -> RationalIsolatingInterval:
    """Compact a reported enclosure outward, retaining exact singletons.

    Exact comparison uses the original internal intervals. Signed integer
    floor/ceiling preserve containment, including negative circle coordinates.
    Keeping singletons preserves the rational-parameter image convention.
    Parameter widths are below 1. Interval evaluation overestimation is at
    most 2**(2d) times the Cayley l1 bound, hence value upper < 10**77 for the
    admitted input frame. 100-place rounding therefore fits 256 digits; the
    circle images are bounded by 5. Exact singletons have separate bounds.
    """
    if lower == upper:
        return _interval(lower, upper)
    low_integer = lower.numerator * _REPORTING_SCALE // lower.denominator
    high_integer = -((-upper.numerator * _REPORTING_SCALE) // upper.denominator)
    return _interval(
        Fraction(low_integer, _REPORTING_SCALE),
        Fraction(high_integer, _REPORTING_SCALE),
    )


def _bounded_parameter_interval(
    polynomial: Any, lower: Fraction, upper: Fraction
) -> tuple[Fraction, Fraction]:
    """Retain one isolated root within the canonical endpoint envelope.

    The primitive squarefree part of the entire derivative has degree <= 16,
    coefficient l1 (hence l2) norm < 10**647, since it divides the primitive
    derivative. Thus separation covers
    roots from different irreducible factors as well.
    Cauchy bounds root magnitude by 10**648; Mignotte's separation bound
    n**(-(n+2)/2)*||f||_2**(1-n) is greater than 10**-9716 (the nonzero
    integer discriminant has magnitude >= 1). See Sasaki--Kako, "An Algebraic
    Method for Separating Close-root Clusters and the Minimum Root
    Separation", p.2, https://www.e.ics.nara-wu.ac.jp/~kako/publications/RootSeparation.pdf.
    Continuation to width <= 10**-10000 and outward rounding on that grid
    preserves the isolated root, since the new width is at most 3e-10000.
    Its endpoint components then need fewer than 10650 decimal digits.
    """
    if max(_digits(lower), _digits(upper)) <= MAX_SUP_NORM_PARAMETER_COMPONENT_DIGITS:
        return lower, upper
    if lower == upper:
        # Rational roots divide the admitted derivative's constant/leading
        # coefficients, so an oversized singleton is a backend contract fault.
        raise _invalid_backend_output("oversized rational critical parameter")
    scale = 10**MAX_SUP_NORM_PARAMETER_DECIMAL_PLACES
    refined_lower, refined_upper = polynomial.sqf_part().refine_root(
        _sympy_rational(lower),
        _sympy_rational(upper),
        eps=sympy.Rational(1, scale),
        fast=True,
    )
    request_checkpoint("compacting unit-circle critical parameter")
    lower = Fraction(int(refined_lower.p), int(refined_lower.q))
    upper = Fraction(int(refined_upper.p), int(refined_upper.q))
    if lower == upper:
        return lower, upper
    return (
        Fraction(lower.numerator * scale // lower.denominator, scale),
        Fraction(-((-upper.numerator * scale) // upper.denominator), scale),
    )


def _admit_sup_norm_polynomial(
    polynomial: GaussianRationalPolynomial,
) -> list[tuple[Fraction, Fraction]]:
    """Admit one input and preflight every derived exact-arithmetic bound.

    Limits are computed before polynomial expansion from the actual scalar
    denominators, Cayley l1 bounds, diagonal cancellation in the derivative,
    Sylvester determinants and Mignotte factor bounds. Transformation scratch,
    derivative isolation, elimination and factoring have separate envelopes.
    """

    if not isinstance(polynomial, GaussianRationalPolynomial):
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="polynomial.unit_circle.sup_norm.polynomial_type",
            message="a bounded Gaussian-rational polynomial in z is required",
        )
    terms = polynomial.terms
    if not terms:
        return [(Fraction(0), Fraction(0))]
    degree = terms[0].exponent
    if degree > MAX_SUP_NORM_DEGREE:
        raise _resource(
            "degree_bound",
            "input polynomial degree exceeds the "
            f"{MAX_SUP_NORM_DEGREE}-degree envelope",
        )
    maximum_digits = 0
    for term in terms:
        maximum_digits = max(
            maximum_digits,
            _digits(term.coefficient.real.as_fraction()),
            _digits(term.coefficient.imaginary.as_fraction()),
        )
    if maximum_digits > MAX_SUP_NORM_INPUT_COMPONENT_DIGITS:
        raise _resource(
            "coefficient_height_bound",
            "input coefficient components exceed the "
            f"{MAX_SUP_NORM_INPUT_COMPONENT_DIGITS}-digit exact-growth bound",
        )
    critical_bound = 2 * degree + 1
    if critical_bound > MAX_SUP_NORM_CRITICAL_ROOTS:
        raise _resource(
            "critical_root_bound",
            "the derivative admits more finite critical points than the envelope",
        )
    comparisons = (critical_bound + 1) * (critical_bound + 2) // 2
    if comparisons > MAX_SUP_NORM_COMPARISONS:
        raise _resource(
            "comparison_bound",
            "the critical-value comparison ledger exceeds its work bound",
        )
    coefficients: list[tuple[Fraction, Fraction]] = [(Fraction(0), Fraction(0))] * (
        degree + 1
    )
    for term in terms:
        coefficients[term.exponent] = (
            term.coefficient.real.as_fraction(),
            term.coefficient.imaginary.as_fraction(),
        )
    bounds = sup_norm_bounds(coefficients)
    for reason, quantity, limit in (
        (
            "derived_height_bound",
            bounds.transform_digits,
            MAX_SUP_NORM_TRANSFORM_COMPONENT_DIGITS,
        ),
        (
            "derivative_height_bound",
            bounds.derivative_digits,
            MAX_SUP_NORM_DERIVATIVE_COMPONENT_DIGITS,
        ),
        (
            "resultant_height_bound",
            max(bounds.resultant_digits, bounds.factor_digits),
            MAX_SUP_NORM_RESULTANT_COMPONENT_DIGITS,
        ),
        ("transform_work_bound", bounds.transform_work, MAX_SUP_NORM_TRANSFORM_WORK),
        ("isolation_work_bound", bounds.isolation_work, MAX_SUP_NORM_ISOLATION_WORK),
        ("resultant_work_bound", bounds.resultant_work, MAX_SUP_NORM_RESULTANT_WORK),
        ("factor_work_bound", bounds.factor_work, MAX_SUP_NORM_FACTOR_WORK),
        (
            "value_analysis_work_bound",
            bounds.value_analysis_work,
            MAX_SUP_NORM_VALUE_ANALYSIS_WORK,
        ),
        (
            "postprocessing_work_bound",
            bounds.postprocessing_work,
            MAX_SUP_NORM_POSTPROCESSING_WORK,
        ),
        (
            "certificate_bound",
            bounds.certificate_scalar_digits,
            MAX_SUP_NORM_CERTIFICATE_SCALAR_DIGITS,
        ),
    ):
        if quantity > limit:
            raise _resource(
                reason, "derived exact work or scalar growth exceeds its owner envelope"
            )
    return coefficients


def _transformed_numerator(
    coefficients: list[tuple[Fraction, Fraction]],
) -> tuple[list[Fraction], list[Fraction]]:
    """Return ascending rational coefficients of ``Q`` and its derivative numerator."""

    n = len(coefficients) - 1
    t = sympy.Symbol("t", real=True)
    expression = 0
    for exponent, (real, imaginary) in enumerate(coefficients):
        if real == 0 and imaginary == 0:
            continue
        value = sympy.Rational(real.numerator, real.denominator) + sympy.I * (
            sympy.Rational(imaginary.numerator, imaginary.denominator)
        )
        expression += (
            value * (1 + sympy.I * t) ** exponent * (1 - sympy.I * t) ** (n - exponent)
        )
    expression = sympy.expand(expression)
    real_part = sympy.expand(sympy.re(expression))
    imaginary_part = sympy.expand(sympy.im(expression))
    numerator = sympy.expand(real_part**2 + imaginary_part**2)
    derivative = sympy.expand(
        sympy.diff(numerator, t) * (1 + t**2) - 2 * n * t * numerator
    )
    return _ascending_fractions(numerator, t), _ascending_fractions(derivative, t)


def _ascending_fractions(expression: Any, symbol: Any) -> list[Fraction]:
    polynomial = sympy.Poly(expression, symbol, domain=sympy.QQ)
    descending = polynomial.all_coeffs()
    if not descending:
        return [Fraction(0)]
    values = [Fraction(int(value.p), int(value.q)) for value in descending]
    values.reverse()
    return values


def _expression(ascending: list[Fraction], symbol: Any) -> Any:
    return sum(
        sympy.Rational(value.numerator, value.denominator) * symbol**exponent
        for exponent, value in enumerate(ascending)
    )


def _sympy_rational(value: Fraction) -> Any:
    return sympy.Rational(value.numerator, value.denominator)


def _interval_horner(
    ascending: list[Fraction], lower: Fraction, upper: Fraction
) -> tuple[Fraction, Fraction]:
    low = high = Fraction(0)
    for coefficient in reversed(ascending):
        products = (
            low * lower,
            low * upper,
            high * lower,
            high * upper,
        )
        low = min(products) + coefficient
        high = max(products) + coefficient
    return low, high


def _divide_interval(
    numerator_low: Fraction,
    numerator_high: Fraction,
    denominator_low: Fraction,
    denominator_high: Fraction,
) -> tuple[Fraction, Fraction]:
    candidates = (
        numerator_low / denominator_low,
        numerator_low / denominator_high,
        numerator_high / denominator_low,
        numerator_high / denominator_high,
    )
    return min(candidates), max(candidates)


def _value_enclosure(
    numerator: list[Fraction],
    denominator_exponent: int,
    lower: Fraction,
    upper: Fraction,
) -> tuple[Fraction, Fraction]:
    numerator_low, numerator_high = _interval_horner(numerator, lower, upper)
    square_low = (
        Fraction(0) if lower <= 0 <= upper else min(lower * lower, upper * upper)
    )
    square_high = max(lower * lower, upper * upper)
    denominator_low = (1 + square_low) ** denominator_exponent
    denominator_high = (1 + square_high) ** denominator_exponent
    return (
        max(Fraction(0), numerator_low) / denominator_high,
        numerator_high / denominator_low,
    )


def _circle_image_enclosure(
    lower: Fraction, upper: Fraction
) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    square_low = (
        Fraction(0) if lower <= 0 <= upper else min(lower * lower, upper * upper)
    )
    square_high = max(lower * lower, upper * upper)
    real_low, real_high = _divide_interval(
        1 - square_high, 1 - square_low, 1 + square_low, 1 + square_high
    )
    imaginary_low, imaginary_high = _divide_interval(
        2 * lower, 2 * upper, 1 + square_low, 1 + square_high
    )
    return real_low, real_high, imaginary_low, imaginary_high


def _refine_isolating_interval(
    polynomial: Any,
    lower: Fraction,
    upper: Fraction,
    target: Fraction,
) -> tuple[Fraction, Fraction]:
    while upper - lower > target:
        middle = (lower + upper) / 2
        middle_sympy = _sympy_rational(middle)
        if polynomial.eval(middle_sympy) == 0:
            return middle, middle
        count = strict_root_count(polynomial, _sympy_rational(lower), middle_sympy)
        if count >= 1:
            upper = middle
        else:
            lower = middle
    return lower, upper


def _root_inside(polynomial: Any, lower: Fraction, upper: Fraction) -> bool:
    if lower == upper:
        return bool(polynomial.eval(_sympy_rational(lower)) == 0)
    return (
        strict_root_count(polynomial, _sympy_rational(lower), _sympy_rational(upper))
        >= 1
    )


def _endpoint_modulus_squared(
    coefficients: list[tuple[Fraction, Fraction]],
) -> Fraction:
    real = Fraction(0)
    imaginary = Fraction(0)
    for exponent, (a, b) in enumerate(coefficients):
        sign = -1 if exponent % 2 else 1
        real += sign * a
        imaginary += sign * b
    return real * real + imaginary * imaginary


def _sqrt_bounds(value: Fraction) -> tuple[Fraction, Fraction]:
    if value <= 0:
        return Fraction(0), Fraction(0)
    scaled = value.numerator * value.denominator
    root = isqrt(scaled)
    lower = Fraction(root, value.denominator)
    if root * root == scaled:
        return lower, lower
    return lower, Fraction(root + 1, value.denominator)


def _rational_algebraic_value(value: Fraction) -> RealAlgebraicValue | None:
    """Return a rational maximum when it fits the shared exact carrier."""

    if _digits(value) > MAX_REAL_ALGEBRAIC_COEFFICIENT_DIGITS:
        return None
    return RealAlgebraicValue._from_admitted_polynomial(
        polynomial=(value.denominator, -value.numerator),
        real_root_index=0,
    )


def _primitive_descending_integers(
    descending: list[Fraction],
) -> tuple[int, ...] | None:
    """Return the primitive positive-leading integer spelling of a factor.

    Returns ``None`` when the spelling leaves the shared real-algebraic
    coefficient envelope.
    """

    denominator = 1
    for value in descending:
        denominator = lcm(denominator, value.denominator)
    integers = [int(value * denominator) for value in descending]
    content = 0
    for coefficient in integers:
        content = gcd(content, coefficient)
    if content == 0:
        return None
    integers = [coefficient // content for coefficient in integers]
    if integers[0] < 0:
        integers = [-coefficient for coefficient in integers]
    if integers[0] <= 0:
        return None
    if any(
        decimal_digit_width(coefficient) > MAX_REAL_ALGEBRAIC_COEFFICIENT_DIGITS
        for coefficient in integers
    ):
        return None
    return tuple(integers)


def _sturm_root_index_below(factor: Any, bound: Fraction) -> int | None:
    """Count the factor's real roots strictly below ``bound`` via Sturm."""

    sequence = factor.sturm()
    if not sequence:
        return None
    point = _sympy_rational(bound)
    negative_signs: list[int] = []
    for term in sequence:
        coefficients = term.all_coeffs()
        if not coefficients:
            continue
        sign = 1 if coefficients[0] > 0 else -1
        if term.degree() % 2:
            sign = -sign
        negative_signs.append(sign)
    bound_signs: list[int] = []
    for term in sequence:
        value = term.eval(point)
        if value == 0:
            continue
        bound_signs.append(1 if value > 0 else -1)
    negative_variations = sum(
        1 for first, second in pairwise(negative_signs) if first != second
    )
    bound_variations = sum(
        1 for first, second in pairwise(bound_signs) if first != second
    )
    return negative_variations - bound_variations


def _flint_resultant_and_factors_impl(
    coprime: Any,
    numerator: list[Fraction],
    degree: int,
    symbol: Any,
) -> tuple[Any, tuple[Any, ...]]:
    """Compute and factor the critical-value resultant through python-flint.

    The input and output stay exact.  SymPy receives only the reconstructed
    univariate result and already-factorized factors for its certified Sturm
    isolation and interval comparison routines.
    """

    from flint import fmpq, fmpq_mpoly_ctx, fmpq_poly

    context = fmpq_mpoly_ctx.get((str(symbol), "y"), "lex")

    def mpoly_from_sympy(poly: Any) -> Any:
        terms: dict[tuple[int, ...], fmpq] = {
            (int(exponents[0]), 0): fmpq(int(coefficient.p), int(coefficient.q))
            for exponents, coefficient in poly.terms()
        }
        return context.from_dict(terms)

    derivative_mpoly = mpoly_from_sympy(coprime)
    numerator_mpoly = context.from_dict(
        {
            (index, 0): fmpq(value.numerator, value.denominator)
            for index, value in enumerate(numerator)
            if value
        }
    )
    t, y = context.gens()
    level_mpoly = y * (1 + t * t) ** degree - numerator_mpoly
    request_checkpoint("before unit-circle FLINT resultant")
    resultant_mpoly = derivative_mpoly.resultant(level_mpoly, str(symbol))
    request_checkpoint("after unit-circle FLINT resultant")
    resultant_terms = resultant_mpoly.to_dict()
    if any(exponents[0] != 0 for exponents in resultant_terms):
        raise _invalid_backend_output(
            "FLINT returned a bivariate critical-value resultant"
        )
    if not resultant_terms:
        raise _invalid_backend_output("critical-value resultant vanished unexpectedly")
    maximum_exponent = max(exponents[1] for exponents in resultant_terms)
    coefficients = [fmpq(0) for _ in range(maximum_exponent + 1)]
    for exponents, coefficient in resultant_terms.items():
        coefficients[exponents[1]] = fmpq(int(coefficient.p), int(coefficient.q))
    resultant_flint = fmpq_poly(coefficients)
    if resultant_flint.is_zero():
        raise _invalid_backend_output("critical-value resultant vanished unexpectedly")
    content, raw_factors = resultant_flint.factor()
    request_checkpoint("after unit-circle FLINT factorization")
    reconstructed = fmpq_poly([content])
    for factor, multiplicity in raw_factors:
        reconstructed *= factor ** int(multiplicity)
    if reconstructed != resultant_flint:
        raise _invalid_backend_output(
            "FLINT returned a non-reconstructing resultant factorization"
        )

    y_symbol = sympy.Symbol("y")
    resultant_expression = sum(
        sympy.Rational(int(coefficient.p), int(coefficient.q)) * y_symbol**index
        for index, coefficient in enumerate(resultant_flint.coeffs())
    )
    resultant_polynomial = sympy.Poly(resultant_expression, y_symbol, domain=sympy.QQ)
    factors: list[Any] = []
    for factor, _multiplicity in raw_factors:
        factor_expression = sum(
            sympy.Rational(int(coefficient.p), int(coefficient.q)) * y_symbol**index
            for index, coefficient in enumerate(factor.coeffs())
        )
        factors.append(sympy.Poly(factor_expression, y_symbol, domain=sympy.QQ))
    return resultant_polynomial, tuple(factors)


def _flint_resultant_and_factors(
    coprime: Any,
    numerator: list[Fraction],
    degree: int,
    symbol: Any,
) -> tuple[Any, tuple[Any, ...]]:
    """Run the exact FLINT adapter with typed backend-failure boundaries."""

    try:
        # Probe the maintained symbols before entering the mathematical
        # adapter.  Import/version failures are initialization failures;
        # everything after this boundary is backend output validation.
        from flint import fmpq, fmpq_mpoly_ctx, fmpq_poly  # noqa: F401
    except (ImportError, ModuleNotFoundError, AttributeError) as exc:
        raise OperationBackendError(BackendFailureReason.INITIALIZATION) from exc
    try:
        return _flint_resultant_and_factors_impl(coprime, numerator, degree, symbol)
    except (
        OperationBackendError,
        OperationExecutionTimeoutError,
        OperationExecutionCancelledError,
    ):
        raise
    except Exception as exc:
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT) from exc


def _exact_critical_maximum(
    resultant_polynomial: Any,
    resultant_factors: tuple[Any, ...],
    value_roots: tuple[tuple[Fraction, Fraction], ...],
    maximum_key: int,
) -> RealAlgebraicValue | None:
    """Identify the maximal critical value as an indexed resultant-factor root.

    Every irreducible factor has simple roots, so the factor carrying the
    maximum is the unique one changing sign strictly inside the maximal
    isolating interval (or vanishing at a singleton).  The resultant
    ``y``-degree is the coprime derivative-numerator degree, which is at most
    twice the input degree: writing ``Q = A**2 + B**2`` with ``deg <= d``,
    the ``t**(2m+1)`` leading terms of ``Q'(1+t**2)`` and ``2*d*t*Q`` cancel
    exactly, so the minimal factor always fits the shared carrier.  Returns
    ``None`` only on theoretical coefficient-height overflow; the certified
    enclosure still carries the complete maximum in that case.
    """

    lower, upper = value_roots[maximum_key]
    candidates: list[Any] = []
    for factor in resultant_factors:
        if factor.degree() < 1:
            continue
        low_value = factor.eval(_sympy_rational(lower))
        if lower == upper:
            if low_value == 0:
                candidates.append(factor)
            continue
        if low_value * factor.eval(_sympy_rational(upper)) < 0:
            candidates.append(factor)
    if len(candidates) != 1:
        return None
    factor = candidates[0]
    degree = factor.degree()
    if degree > MAX_REAL_ALGEBRAIC_DEGREE:
        # Defensive only: the cancellation noted above keeps every minimal
        # factor within twice the input degree (hence at most 16 here).
        return None
    descending = [
        Fraction(int(coefficient.p), int(coefficient.q))
        for coefficient in factor.all_coeffs()
    ]
    spelling = _primitive_descending_integers(descending)
    if spelling is None or len(spelling) - 1 != degree:
        return None
    if degree == 1:
        return RealAlgebraicValue._from_admitted_polynomial(
            polynomial=spelling, real_root_index=0
        )
    if lower == upper:
        # An irreducible factor of degree two or more has no rational root.
        return None
    if factor.eval(_sympy_rational(lower)) == 0:
        return None
    index = _sturm_root_index_below(factor, lower)
    if index is None or not 0 <= index < degree:
        return None
    return RealAlgebraicValue._from_admitted_polynomial(
        polynomial=spelling, real_root_index=index
    )


def unit_circle_sup_norm_squared(  # noqa: C901
    polynomial: GaussianRationalPolynomial,
) -> UnitCircleSupNormSquaredResult:
    """Return a certified enclosure of ``max_{|z|=1} |P(z)|^2`` and its ledger."""

    execution = current_request_execution()
    if execution is None:
        with request_execution(time.monotonic()):
            return unit_circle_sup_norm_squared(polynomial)
    started = execution.started_at
    deadline = started + 120.0
    if execution is not None and execution.deadline is not None:
        deadline = min(deadline, execution.deadline)

    def checkpoint() -> None:
        request_checkpoint("computing unit-circle supremum norm")
        if time.monotonic() >= deadline:
            raise OperationExecutionTimeoutError(
                "unit-circle supremum norm deadline expired"
            )

    bind_request_deadline(deadline)
    checkpoint()
    coefficients = _admit_sup_norm_polynomial(polynomial)
    degree = len(coefficients) - 1
    numerator, derivative = _transformed_numerator(coefficients)
    endpoint_value = _endpoint_modulus_squared(coefficients)
    checkpoint()

    numerator_canonical = tuple(
        CanonicalRational(num=value.numerator, den=value.denominator)
        for value in numerator
    )
    derivative_canonical = tuple(
        CanonicalRational(num=value.numerator, den=value.denominator)
        for value in derivative
    )

    if all(value == 0 for value in derivative):
        constant = numerator[0] if numerator else Fraction(0)
        enclosure = _rational_interval(constant)
        lower, upper = _sqrt_bounds(constant)
        result = UnitCircleSupNormSquaredResult._from_kernel(
            polynomial=polynomial,
            degree=degree,
            transformed_numerator=numerator_canonical,
            denominator_exponent=degree,
            derivative_numerator=derivative_canonical,
            critical_points=(),
            endpoint_minus_one_value=CanonicalRational(
                num=endpoint_value.numerator, den=endpoint_value.denominator
            ),
            endpoint_minus_one_is_maximizer=True,
            maximizing_status="FULL_CIRCLE",
            sup_norm_squared_enclosure=enclosure,
            sup_norm_squared_exact=_rational_algebraic_value(constant),
            sup_norm_enclosure=_reported_interval(lower, upper),
        )
        checkpoint()
        return result

    symbol = sympy.Symbol("t", real=True)
    derivative_polynomial = sympy.Poly(
        _expression(derivative, symbol), symbol, domain=sympy.QQ
    )
    numerator_expression = _expression(numerator, symbol)
    isolating = tuple(
        derivative_polynomial.intervals(
            eps=_sympy_rational(_REFINEMENT_TARGETS[-1]), fast=True
        )
    )
    checkpoint()

    # The resultant of the derivative numerator and the level relation defines
    # one rational polynomial whose real roots are exactly the critical values,
    # which gives exact value identity and order without algebraic comparison.
    coprime = derivative_polynomial
    common = sympy.gcd(coprime, sympy.Poly(1 + symbol**2, symbol, domain=sympy.QQ))
    while common.degree() > 0:
        coprime = sympy.div(coprime, common)[0]
        common = sympy.gcd(coprime, sympy.Poly(1 + symbol**2, symbol, domain=sympy.QQ))
    resultant_polynomial, resultant_factors = _flint_resultant_and_factors(
        coprime,
        numerator,
        degree,
        symbol,
    )
    value_roots = tuple(
        (Fraction(int(lower.p), int(lower.q)), Fraction(int(upper.p), int(upper.q)))
        for (lower, upper), _multiplicity in resultant_polynomial.intervals(fast=True)
    )
    checkpoint()

    level_cache: dict[Fraction, Any] = {}

    def rational_level_gcd(value: Fraction) -> Any | None:
        if value in level_cache:
            return level_cache[value]
        level = sympy.expand(
            numerator_expression - _sympy_rational(value) * (1 + symbol**2) ** degree
        )
        divisor = sympy.gcd(
            derivative_polynomial, sympy.Poly(level, symbol, domain=sympy.QQ)
        )
        result = divisor if divisor.degree() >= 1 else None
        level_cache[value] = result
        return result

    entries: list[
        tuple[int, tuple[Fraction, Fraction], tuple[Fraction, Fraction], int]
    ] = []
    for root_index, ((root_lower, root_upper), _multiplicity) in enumerate(isolating):
        lower = Fraction(int(root_lower.p), int(root_lower.q))
        upper = Fraction(int(root_upper.p), int(root_upper.q))
        lower, upper = _bounded_parameter_interval(derivative_polynomial, lower, upper)
        key: int | None = None
        for target in _REFINEMENT_TARGETS:
            lower, upper = _refine_isolating_interval(
                derivative_polynomial, lower, upper, target
            )
            value_low, value_high = _value_enclosure(numerator, degree, lower, upper)
            candidates = [
                index
                for index, (value_lower, value_upper) in enumerate(value_roots)
                if value_high >= value_lower and value_low <= value_upper
            ]
            if len(candidates) == 1:
                key = candidates[0]
                break
            for index in candidates:
                singleton_lower, singleton_upper = value_roots[index]
                if singleton_lower != singleton_upper:
                    continue
                divisor = rational_level_gcd(singleton_lower)
                if divisor is not None and _root_inside(divisor, lower, upper):
                    key = index
                    break
            if key is not None:
                break
        if key is None:
            raise _resource(
                "value_identification",
                "a critical value could not be identified with a rational level",
            )
        if value_roots[key] == (Fraction(0), Fraction(0)):
            # Identity with zero was already established above. Retain it
            # exactly: the nonnegative interval bound clips at zero, which
            # would exclude this value from an OPEN nonsingleton enclosure.
            value_low = value_high = Fraction(0)
        else:
            value_low, value_high = _value_enclosure(numerator, degree, lower, upper)
        entries.append((root_index, (lower, upper), (value_low, value_high), key))
    checkpoint()

    distinct_keys = sorted({entry[3] for entry in entries})
    greater_keys = {
        key: sum(1 for other in distinct_keys if other > key) for key in distinct_keys
    }
    all_endpoint_comparisons = _compare_rational_to_value_roots(
        resultant_polynomial, resultant_factors, value_roots, endpoint_value
    )
    endpoint_comparison = {key: all_endpoint_comparisons[key] for key in distinct_keys}
    endpoint_rank = sum(1 for key in distinct_keys if endpoint_comparison[key] > 0)

    critical_points: list[UnitCircleCriticalPoint] = []
    for root_index, (lower, upper), (value_low, value_high), key in entries:
        rank = greater_keys[key] + (1 if endpoint_comparison[key] < 0 else 0)
        real_low, real_high, imaginary_low, imaginary_high = _circle_image_enclosure(
            lower, upper
        )
        critical_points.append(
            UnitCircleCriticalPoint(
                root_index=root_index,
                parameter=_interval(lower, upper),
                value=_reported_interval(value_low, value_high),
                z_real=_reported_interval(real_low, real_high),
                z_imaginary=_reported_interval(imaginary_low, imaginary_high),
                comparison_rank=rank,
                is_maximizer=rank == 0,
            )
        )

    maximum_key = distinct_keys[-1]
    critical_is_maximizer = greater_keys[maximum_key] == 0 and (
        endpoint_comparison[maximum_key] >= 0
    )
    endpoint_is_maximizer = endpoint_rank == 0
    if endpoint_is_maximizer and critical_is_maximizer:
        status = "CRITICAL_AND_ENDPOINT"
    elif endpoint_is_maximizer:
        status = "ENDPOINT"
    else:
        status = "ISOLATED_CRITICAL"

    if status == "ENDPOINT" or status == "CRITICAL_AND_ENDPOINT":
        squared_low = squared_high = endpoint_value
    else:
        maximizing_values = [entry[2] for entry in entries if entry[3] == maximum_key]
        squared_low = min(value[0] for value in maximizing_values)
        squared_high = max(value[1] for value in maximizing_values)
    squared_enclosure = _reported_interval(squared_low, squared_high)
    sup_norm_low, _ = _sqrt_bounds(squared_enclosure.lower.as_fraction())
    _, sup_norm_high = _sqrt_bounds(squared_enclosure.upper.as_fraction())

    checkpoint()
    if endpoint_is_maximizer:
        exact_maximum: RealAlgebraicValue | None = _rational_algebraic_value(
            endpoint_value
        )
    else:
        exact_maximum = _exact_critical_maximum(
            resultant_polynomial, resultant_factors, value_roots, maximum_key
        )

    result = UnitCircleSupNormSquaredResult._from_kernel(
        polynomial=polynomial,
        degree=degree,
        transformed_numerator=numerator_canonical,
        denominator_exponent=degree,
        derivative_numerator=derivative_canonical,
        critical_points=tuple(critical_points),
        endpoint_minus_one_value=CanonicalRational(
            num=endpoint_value.numerator, den=endpoint_value.denominator
        ),
        endpoint_minus_one_is_maximizer=endpoint_is_maximizer,
        maximizing_status=status,
        sup_norm_squared_enclosure=squared_enclosure,
        sup_norm_squared_exact=exact_maximum,
        sup_norm_enclosure=_reported_interval(sup_norm_low, sup_norm_high),
    )
    checkpoint()
    return result


def _compare_rational_to_value_roots(
    polynomial: Any,
    factors: tuple[Any, ...],
    root_intervals: tuple[tuple[Fraction, Fraction], ...],
    value: Fraction,
) -> tuple[int, ...]:
    """Order one rational endpoint against sorted resultant-root intervals.

    The isolated intervals are disjoint and ordered.  Binary search locates
    the sole interval that can contain ``value``.  When it does, an exact sign
    change in its irreducible resultant factor determines which side of the
    root contains ``value`` without a Sturm count on the full resultant.
    """

    if not root_intervals:
        return ()
    position = bisect_left([upper for _lower, upper in root_intervals], value)
    comparisons = [-1] * min(position, len(root_intervals))
    if position == len(root_intervals):
        return tuple(comparisons)
    lower, upper = root_intervals[position]
    if value < lower:
        return (*comparisons, *(1 for _ in root_intervals[position:]))
    if lower == upper:
        comparison = 0 if value == lower else (1 if lower > value else -1)
    elif value == lower:
        comparison = 1
    elif value == upper:
        comparison = -1
    elif polynomial.eval(_sympy_rational(value)) == 0:
        comparison = 0
    else:
        lower_point = _sympy_rational(lower)
        upper_point = _sympy_rational(upper)
        value_point = _sympy_rational(value)
        carrying_factors = tuple(
            factor
            for factor in factors
            if factor.eval(lower_point) * factor.eval(upper_point) < 0
        )
        if len(carrying_factors) == 1:
            factor = carrying_factors[0]
            value_sign = factor.eval(value_point)
            comparison = (
                0
                if value_sign == 0
                else (-1 if factor.eval(lower_point) * value_sign < 0 else 1)
            )
        else:
            count = strict_root_count(polynomial, lower_point, value_point)
            comparison = -1 if count else 1
    return (
        *comparisons,
        comparison,
        *(1 for _ in root_intervals[position + 1 :]),
    )


def verify_unit_circle_sup_norm_squared(
    claim: UnitCircleSupNormSquaredResult,
) -> bool:
    """Check a claimed supremum-norm certificate by replaying the kernel.

    The admitted kernel is deterministic and canonical, so a claim is accepted
    exactly when it equals the recomputed result.  This rejects forged
    aggregate fields (the enclosures, ``maximizing_status``, comparison ranks,
    ``is_maximizer`` flags, and the exact value) in addition to mismatched
    defining data; a weakened claim that drops the exact value is rejected too.
    """

    if not isinstance(claim, UnitCircleSupNormSquaredResult):
        return False
    if not isinstance(claim.polynomial, GaussianRationalPolynomial):
        return False
    try:
        recomputed = unit_circle_sup_norm_squared(claim.polynomial)
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False
    return recomputed == claim
