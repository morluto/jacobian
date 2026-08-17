"""Exact elementary polynomial operations backed by SymPy ``Poly`` APIs."""

from __future__ import annotations

from functools import cache
from typing import Any

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.math import polynomials
from jacobian.math.polynomials._conversions import (
    rational_from_sympy,
    rational_polynomial_from_sympy,
    rational_polynomial_to_sympy,
)
from jacobian.math.polynomials._models import (
    IntegerPolynomial,
    IntegerPolynomialCompositionRequest,
    IntegerPolynomialCompositionResult,
    IntegerPolynomialContentResult,
    IntegerPolynomialEvaluationRequest,
    IntegerPolynomialEvaluationResult,
    IntegerPolynomialGcdResult,
    IntegerPolynomialPairRequest,
    IntegerPolynomialPrimitivePartResult,
    IntegerPolynomialRequest,
    IntegerPolynomialShiftRequest,
    IntegerPolynomialShiftResult,
    RationalFunctionRequest,
    RationalPartialFractionResult,
    RationalPartialFractionTerm,
    RationalPolynomialDerivativeResult,
    RationalPolynomialDivisionRequest,
    RationalPolynomialDivisionResult,
    RationalPolynomialEvaluationRequest,
    RationalPolynomialEvaluationResult,
    RationalPolynomialIntegralResult,
    RationalPolynomialRequest,
)


@cache
def _x() -> Any:
    """Load the canonical integer-polynomial indeterminate on first invocation."""

    from sympy import Symbol

    return Symbol("x")


def _integer_poly(polynomial: IntegerPolynomial) -> Any:
    from sympy import Poly

    return Poly.from_list(
        [
            parse_canonical_integer(coefficient)
            for coefficient in polynomial.coefficients
        ],
        _x(),
        domain="ZZ",
    )


def _integer_wire(polynomial: Any) -> IntegerPolynomial:
    return IntegerPolynomial(
        coefficients=tuple(
            format_canonical_integer(int(coefficient))
            for coefficient in polynomial.all_coeffs()
        )
    )


def integer_polynomial_gcd(
    request: IntegerPolynomialPairRequest,
) -> IntegerPolynomialGcdResult:
    left = _integer_poly(request.left)
    right = _integer_poly(request.right)
    gcd = left.gcd(right)
    return IntegerPolynomialGcdResult(
        gcd=_integer_wire(gcd),
        left_content=format_canonical_integer(int(left.content())),
        right_content=format_canonical_integer(int(right.content())),
        gcd_content=format_canonical_integer(int(gcd.content())),
    )


def integer_polynomial_content(
    request: IntegerPolynomialRequest,
) -> IntegerPolynomialContentResult:
    return IntegerPolynomialContentResult(
        content=format_canonical_integer(
            int(_integer_poly(request.polynomial).content())
        )
    )


def integer_polynomial_primitive_part(
    request: IntegerPolynomialRequest,
) -> IntegerPolynomialPrimitivePartResult:
    source = _integer_poly(request.polynomial)
    content, primitive = source.primitive()
    reconstructed = primitive.mul_ground(content)
    return IntegerPolynomialPrimitivePartResult(
        content=format_canonical_integer(int(content)),
        primitive_part=_integer_wire(primitive),
        reconstruction=_integer_wire(reconstructed),
    )


def integer_polynomial_evaluate(
    request: IntegerPolynomialEvaluationRequest,
) -> IntegerPolynomialEvaluationResult:
    point = parse_canonical_integer(request.point)
    value = _integer_poly(request.polynomial).eval(point)
    return IntegerPolynomialEvaluationResult(
        point=request.point,
        value=format_canonical_integer(int(value)),
    )


def integer_polynomial_compose(
    request: IntegerPolynomialCompositionRequest,
) -> IntegerPolynomialCompositionResult:
    composition = _integer_poly(request.outer).compose(_integer_poly(request.inner))
    return IntegerPolynomialCompositionResult(composition=_integer_wire(composition))


def integer_polynomial_shift(
    request: IntegerPolynomialShiftRequest,
) -> IntegerPolynomialShiftResult:
    """Compute ``p(x + a)`` using SymPy's exact dense shift."""
    shifted = _integer_poly(request.polynomial).shift(request.shift)
    return IntegerPolynomialShiftResult(
        shift=request.shift,
        shifted=_integer_wire(shifted),
    )


def rational_polynomial_division(
    request: RationalPolynomialDivisionRequest,
) -> RationalPolynomialDivisionResult:
    left = rational_polynomial_to_sympy(request.left)
    right = rational_polynomial_to_sympy(request.right)
    quotient, remainder, reconstruction = polynomials.divide(left, right)
    variables = request.left.variables
    return RationalPolynomialDivisionResult(
        quotient=rational_polynomial_from_sympy(quotient, variables),
        remainder=rational_polynomial_from_sympy(remainder, variables),
        reconstruction=rational_polynomial_from_sympy(reconstruction, variables),
    )


def rational_polynomial_evaluate(
    request: RationalPolynomialEvaluationRequest,
) -> RationalPolynomialEvaluationResult:
    point = request.point.as_fraction()
    from sympy import Rational

    value = polynomials.evaluate(
        rational_polynomial_to_sympy(request.polynomial),
        Rational(point.numerator, point.denominator),
    )
    return RationalPolynomialEvaluationResult(
        point=request.point,
        value=rational_from_sympy(value),
    )


def rational_polynomial_derivative(
    request: RationalPolynomialRequest,
) -> RationalPolynomialDerivativeResult:
    return RationalPolynomialDerivativeResult(
        derivative=rational_polynomial_from_sympy(
            polynomials.derivative(rational_polynomial_to_sympy(request.polynomial)),
            request.polynomial.variables,
        )
    )


def rational_polynomial_integral(
    request: RationalPolynomialRequest,
) -> RationalPolynomialIntegralResult:
    return RationalPolynomialIntegralResult(
        antiderivative=rational_polynomial_from_sympy(
            polynomials.integral(rational_polynomial_to_sympy(request.polynomial)),
            request.polynomial.variables,
        )
    )


def _partial_fraction_term(
    numerator: Any,
    denominator: Any,
    generator: Any,
    variables: tuple[str, ...],
) -> RationalPartialFractionTerm:
    from sympy import Poly

    denominator_poly = Poly(denominator, generator, domain="QQ")
    denominator_coefficient, factors = denominator_poly.factor_list()
    if len(factors) != 1:
        raise ValueError("SymPy returned a non-atomic partial-fraction denominator")
    denominator_factor, exponent = factors[0]
    leading_coefficient = denominator_factor.LC()
    monic_factor = denominator_factor.monic()
    scale = denominator_coefficient * leading_coefficient**exponent
    normalized_numerator = Poly(numerator / scale, generator, domain="QQ")
    return RationalPartialFractionTerm(
        numerator=rational_polynomial_from_sympy(normalized_numerator, variables),
        denominator_factor=rational_polynomial_from_sympy(monic_factor, variables),
        denominator_exponent=int(exponent),
    )


def _partial_fraction_sort_key(
    term: RationalPartialFractionTerm,
) -> tuple[tuple[tuple[tuple[int, ...], int, int], ...], int]:
    factor_terms: list[tuple[tuple[int, ...], int, int]] = []
    for factor_term in term.denominator_factor.polynomial.terms:
        coefficient = factor_term.coefficient.as_fraction()
        factor_terms.append(
            (
                factor_term.exponents,
                coefficient.numerator,
                coefficient.denominator,
            )
        )
    return tuple(factor_terms), term.denominator_exponent


def rational_partial_fraction_decomposition(
    request: RationalFunctionRequest,
) -> RationalPartialFractionResult:
    from sympy import Add, Poly, cancel, fraction, together

    variables = request.numerator.variables
    numerator_polynomial = rational_polynomial_to_sympy(request.numerator)
    denominator_polynomial = rational_polynomial_to_sympy(request.denominator)
    generator = numerator_polynomial.gens[0]
    source = cancel(numerator_polynomial.as_expr() / denominator_polynomial.as_expr())
    decomposition = polynomials.partial_fractions(source, generator)
    polynomial_part = Poly(0, generator, domain="QQ")
    proper_terms: list[RationalPartialFractionTerm] = []
    for summand in Add.make_args(decomposition):
        numerator, denominator = fraction(cancel(summand))
        denominator_poly = Poly(denominator, generator, domain="QQ")
        if denominator_poly.degree() == 0:
            polynomial_part += Poly(
                numerator / denominator_poly.LC(),
                generator,
                domain="QQ",
            )
        else:
            proper_terms.append(
                _partial_fraction_term(
                    numerator,
                    denominator,
                    generator,
                    variables,
                )
            )

    reconstructed = cancel(together(decomposition))
    reconstructed_numerator, reconstructed_denominator = fraction(reconstructed)
    denominator_poly = Poly(reconstructed_denominator, generator, domain="QQ")
    denominator_lead = denominator_poly.LC()
    normalized_numerator = Poly(
        reconstructed_numerator / denominator_lead,
        generator,
        domain="QQ",
    )
    normalized_denominator = denominator_poly.monic()
    return RationalPartialFractionResult(
        polynomial_part=rational_polynomial_from_sympy(polynomial_part, variables),
        terms=tuple(sorted(proper_terms, key=_partial_fraction_sort_key)),
        reconstruction_numerator=rational_polynomial_from_sympy(
            normalized_numerator, variables
        ),
        reconstruction_denominator=rational_polynomial_from_sympy(
            normalized_denominator, variables
        ),
    )


__all__ = [
    "integer_polynomial_compose",
    "integer_polynomial_content",
    "integer_polynomial_evaluate",
    "integer_polynomial_gcd",
    "integer_polynomial_primitive_part",
    "rational_partial_fraction_decomposition",
    "rational_polynomial_derivative",
    "rational_polynomial_division",
    "rational_polynomial_evaluate",
    "rational_polynomial_integral",
]
