"""Exact elementary polynomial operations backed by SymPy ``Poly`` APIs."""

from __future__ import annotations

from functools import cache
from typing import Any

from pydantic_core import PydanticCustomError

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS
from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math import polynomials
from jacobian.math.polynomials._conversions import (
    rational_from_sympy,
    rational_polynomial_from_sympy,
    rational_polynomial_to_sympy,
)
from jacobian.math.polynomials._models import (
    _MAX_ELEMENTARY_DEGREE,
    _MAX_GCD_TERMS,
    _MAX_INTEGER_COEFFICIENT_DIGITS,
    _MAX_INVARIANT_TERMS,
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
    _validation_error,
)
from jacobian.math.polynomials.values import (
    rational_evaluation_component_digit_bounds,
    require_polynomial_budget,
)


def _run_admission(admission: Any) -> None:
    try:
        admission()
    except OperationDomainValidationError:
        raise
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=(), code=exc.type, message=exc.message()
        ) from exc
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=(), code="polynomial.admission", message=str(exc)
        ) from exc


def _admit_integer(polynomial: IntegerPolynomial) -> None:
    if len(polynomial.coefficients) > _MAX_ELEMENTARY_DEGREE + 1:
        raise _validation_error(
            f"integer polynomial exceeds the degree-{_MAX_ELEMENTARY_DEGREE} operation budget"
        )
    if any(
        len(coefficient.lstrip("-")) > _MAX_INTEGER_COEFFICIENT_DIGITS
        for coefficient in polynomial.coefficients
    ):
        raise _validation_error("integer coefficient exceeds the decimal-digit budget")


def _admit_integer_pair(request: IntegerPolynomialPairRequest) -> None:
    _admit_integer(request.left)
    _admit_integer(request.right)


def _admit_integer_evaluation(request: IntegerPolynomialEvaluationRequest) -> None:
    _admit_integer(request.polynomial)
    if len(request.point.lstrip("-")) > _MAX_INTEGER_COEFFICIENT_DIGITS:
        raise _validation_error("evaluation point exceeds the decimal-digit budget")


def _admit_integer_composition(request: IntegerPolynomialCompositionRequest) -> None:
    _admit_integer(request.outer)
    _admit_integer(request.inner)
    if (len(request.outer.coefficients) - 1) * (
        len(request.inner.coefficients) - 1
    ) > _MAX_ELEMENTARY_DEGREE:
        raise _validation_error(
            f"composition exceeds the degree-{_MAX_ELEMENTARY_DEGREE} output budget"
        )


def _admit_rational(request: RationalPolynomialRequest) -> None:
    if len(request.polynomial.variables) != 1:
        raise _validation_error("elementary polynomial operations require one variable")
    require_polynomial_budget(
        request.polynomial,
        maximum_terms=_MAX_GCD_TERMS,
        maximum_exponent=_MAX_ELEMENTARY_DEGREE,
    )


def _admit_rational_evaluation(
    request: RationalPolynomialEvaluationRequest,
) -> None:
    _admit_rational(request)
    numerator_digits, denominator_digits = rational_evaluation_component_digit_bounds(
        request.polynomial,
        (request.point,),
    )
    if max(numerator_digits, denominator_digits) > MAX_CANONICAL_RATIONAL_DIGITS:
        raise OperationDomainValidationError(
            location=("polynomial", "point"),
            code="polynomial.evaluation_result_exceeds_component_bound",
            message=(
                "exact evaluation exceeds the "
                f"{MAX_CANONICAL_RATIONAL_DIGITS}-digit rational component bound"
            ),
        )


def _admit_division(request: RationalPolynomialDivisionRequest) -> None:
    if len(request.left.variables) != 1:
        raise _validation_error("polynomial division requires one variable")
    if not request.right.polynomial.terms:
        raise _validation_error("divisor polynomial must be nonzero")
    for polynomial in (request.left, request.right):
        require_polynomial_budget(
            polynomial,
            maximum_terms=_MAX_GCD_TERMS,
            maximum_exponent=_MAX_ELEMENTARY_DEGREE,
        )


def _admit_partial_fractions(request: RationalFunctionRequest) -> None:
    if request.numerator.variables != request.denominator.variables:
        raise _validation_error("numerator and denominator must use the same ring")
    if len(request.numerator.variables) != 1:
        raise _validation_error("partial fractions require one variable")
    if not request.denominator.polynomial.terms:
        raise _validation_error("denominator polynomial must be nonzero")
    for polynomial in (request.numerator, request.denominator):
        require_polynomial_budget(
            polynomial,
            maximum_terms=_MAX_INVARIANT_TERMS,
            maximum_exponent=_MAX_ELEMENTARY_DEGREE,
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
    _run_admission(lambda: _admit_integer_pair(request))
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
    _run_admission(lambda: _admit_integer(request.polynomial))
    return IntegerPolynomialContentResult(
        content=format_canonical_integer(
            int(_integer_poly(request.polynomial).content())
        )
    )


def integer_polynomial_primitive_part(
    request: IntegerPolynomialRequest,
) -> IntegerPolynomialPrimitivePartResult:
    _run_admission(lambda: _admit_integer(request.polynomial))
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
    _run_admission(lambda: _admit_integer_evaluation(request))
    point = parse_canonical_integer(request.point)
    value = _integer_poly(request.polynomial).eval(point)
    return IntegerPolynomialEvaluationResult(
        point=request.point,
        value=format_canonical_integer(int(value)),
    )


def integer_polynomial_compose(
    request: IntegerPolynomialCompositionRequest,
) -> IntegerPolynomialCompositionResult:
    _run_admission(lambda: _admit_integer_composition(request))
    composition = _integer_poly(request.outer).compose(_integer_poly(request.inner))
    return IntegerPolynomialCompositionResult(composition=_integer_wire(composition))


def integer_polynomial_shift(
    request: IntegerPolynomialShiftRequest,
) -> IntegerPolynomialShiftResult:
    """Compute ``p(x + a)`` using SymPy's exact dense shift."""
    _run_admission(lambda: _admit_integer(request.polynomial))
    shifted = _integer_poly(request.polynomial).shift(request.shift)
    return IntegerPolynomialShiftResult(
        shift=request.shift,
        shifted=_integer_wire(shifted),
    )


def rational_polynomial_division(
    request: RationalPolynomialDivisionRequest,
) -> RationalPolynomialDivisionResult:
    _run_admission(lambda: _admit_division(request))
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
    _run_admission(lambda: _admit_rational_evaluation(request))
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
    _run_admission(lambda: _admit_rational(request))
    return RationalPolynomialDerivativeResult(
        derivative=rational_polynomial_from_sympy(
            polynomials.derivative(rational_polynomial_to_sympy(request.polynomial)),
            request.polynomial.variables,
        )
    )


def rational_polynomial_integral(
    request: RationalPolynomialRequest,
) -> RationalPolynomialIntegralResult:
    _run_admission(lambda: _admit_rational(request))
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
    _run_admission(lambda: _admit_partial_fractions(request))
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
