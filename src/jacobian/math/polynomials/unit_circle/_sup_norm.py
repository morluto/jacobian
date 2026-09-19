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
from fractions import Fraction
from itertools import pairwise
from math import gcd, isqrt, lcm
from typing import Any

import sympy

from jacobian._exact import CanonicalRational
from jacobian._execution import (
    BackendFailureReason,
    OperationBackendError,
    OperationExecutionTimeoutError,
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
)
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
from jacobian.math.polynomials.unit_circle._sup_norm_models import (
    MAX_SUP_NORM_CERTIFICATE_BYTES,
    MAX_SUP_NORM_COMPARISONS,
    MAX_SUP_NORM_CRITICAL_ROOTS,
    MAX_SUP_NORM_DEGREE,
    MAX_SUP_NORM_DERIVATIVE_COMPONENT_DIGITS,
    MAX_SUP_NORM_INPUT_COMPONENT_DIGITS,
    MAX_SUP_NORM_ISOLATION_WORK,
    GaussianRationalPolynomial,
    UnitCircleCriticalPoint,
    UnitCircleSupNormSquaredResult,
)

__all__ = [
    "unit_circle_sup_norm_squared",
    "verify_unit_circle_sup_norm_squared",
]

# Rational endpoints of the reported enclosures are refined until the
# parameter width is below this target, which keeps every enclosure well
# inside the shared 256-digit canonical-rational envelope.
_REFINEMENT_TARGETS = (Fraction(1, 10**40), Fraction(1, 10**70), Fraction(1, 10**100))


def _resource(reason: str, message: str) -> OperationResourceAdmissionError:
    return OperationResourceAdmissionError(
        location=("polynomial",),
        code=f"polynomial.unit_circle.sup_norm.{reason}",
        message=message,
    )


def _invalid_backend_output(message: str) -> OperationBackendError:
    return OperationBackendError(BackendFailureReason.INVALID_OUTPUT)


def _digits(value: Fraction) -> int:
    return max(len(str(abs(value.numerator))), len(str(value.denominator)))


def _interval(lower: Fraction, upper: Fraction) -> RationalIsolatingInterval:
    return RationalIsolatingInterval(
        lower=CanonicalRational(num=lower.numerator, den=lower.denominator),
        upper=CanonicalRational(num=upper.numerator, den=upper.denominator),
        interval_type="SINGLETON" if lower == upper else "OPEN",
    )


def _rational_interval(value: Fraction) -> RationalIsolatingInterval:
    return _interval(value, value)


def _admit_sup_norm_polynomial(
    polynomial: GaussianRationalPolynomial,
) -> list[tuple[Fraction, Fraction]]:
    """Admit one input and preflight every derived exact-arithmetic bound.

    All limits are computed from the compressed input before any polynomial
    expansion, root isolation, or resultant construction.  The derived
    numerator/derivative height is bounded by ``4H + 8n + 32`` decimal
    digits, which dominates the binomial and product expansion of an ``H``
    digit, degree ``n`` Gaussian-rational input.
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
    derived_digits = 4 * maximum_digits + 8 * degree + 32
    if derived_digits > MAX_SUP_NORM_DERIVATIVE_COMPONENT_DIGITS:
        raise _resource(
            "derived_height_bound",
            "transformed numerator or derivative exceeds the "
            f"{MAX_SUP_NORM_DERIVATIVE_COMPONENT_DIGITS}-digit bound",
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
    isolation_work = (critical_bound + 1) ** 3 * (derived_digits + 64)
    if isolation_work > MAX_SUP_NORM_ISOLATION_WORK:
        raise _resource(
            "isolation_work_bound",
            "derived exact isolation work exceeds the measured backend envelope",
        )
    certificate_bytes = (critical_bound + 1) * 16 * derived_digits
    if certificate_bytes > MAX_SUP_NORM_CERTIFICATE_BYTES:
        raise _resource(
            "certificate_bound",
            "the retained critical-point certificate exceeds its byte bound",
        )
    coefficients: list[tuple[Fraction, Fraction]] = [(Fraction(0), Fraction(0))] * (
        degree + 1
    )
    for term in terms:
        coefficients[term.exponent] = (
            term.coefficient.real.as_fraction(),
            term.coefficient.imaginary.as_fraction(),
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


def _rational_algebraic_value(value: Fraction) -> RealAlgebraicValue:
    """Return the degree-one carrier of a rational maximum."""

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
        len(str(abs(coefficient))) > MAX_REAL_ALGEBRAIC_COEFFICIENT_DIGITS
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
    resultant_mpoly = derivative_mpoly.resultant(level_mpoly, str(symbol))
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
    except OperationBackendError:
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
    started = execution.started_at if execution is not None else time.monotonic()
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
        return UnitCircleSupNormSquaredResult._from_kernel(
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
            sup_norm_enclosure=_interval(lower, upper),
        )

    symbol = sympy.Symbol("t", real=True)
    derivative_polynomial = sympy.Poly(
        _expression(derivative, symbol), symbol, domain=sympy.QQ
    )
    numerator_expression = _expression(numerator, symbol)
    isolating = tuple(derivative_polynomial.intervals())
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
        for (lower, upper), _multiplicity in resultant_polynomial.intervals()
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
        value_low, value_high = _value_enclosure(numerator, degree, lower, upper)
        entries.append((root_index, (lower, upper), (value_low, value_high), key))
    checkpoint()

    distinct_keys = sorted({entry[3] for entry in entries})
    greater_keys = {
        key: sum(1 for other in distinct_keys if other > key) for key in distinct_keys
    }
    endpoint_comparison = {
        key: _compare_rational_to_value_root(
            resultant_polynomial, value_roots[key], endpoint_value
        )
        for key in distinct_keys
    }
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
                value=_interval(value_low, value_high),
                z_real=_interval(real_low, real_high),
                z_imaginary=_interval(imaginary_low, imaginary_high),
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
    sup_norm_low, _ = _sqrt_bounds(squared_low)
    _, sup_norm_high = _sqrt_bounds(squared_high)

    checkpoint()
    if endpoint_is_maximizer:
        exact_maximum: RealAlgebraicValue | None = _rational_algebraic_value(
            endpoint_value
        )
    else:
        exact_maximum = _exact_critical_maximum(
            resultant_polynomial, resultant_factors, value_roots, maximum_key
        )

    return UnitCircleSupNormSquaredResult._from_kernel(
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
        sup_norm_squared_enclosure=_interval(squared_low, squared_high),
        sup_norm_squared_exact=exact_maximum,
        sup_norm_enclosure=_interval(sup_norm_low, sup_norm_high),
    )


def _compare_rational_to_value_root(
    polynomial: Any,
    root_interval: tuple[Fraction, Fraction],
    value: Fraction,
) -> int:
    """Return the sign of ``critical_value - value`` for one resultant root."""

    lower, upper = root_interval
    if lower == upper:
        if value == lower:
            return 0
        return 1 if lower > value else -1
    if value <= lower:
        return 1
    if value >= upper:
        return -1
    if polynomial.eval(_sympy_rational(value)) == 0:
        return 0
    count = strict_root_count(
        polynomial, _sympy_rational(lower), _sympy_rational(value)
    )
    return -1 if count else 1


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
