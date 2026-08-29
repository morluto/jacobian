"""Exact elementary polynomial operations backed by SymPy ``Poly`` APIs."""

from __future__ import annotations

from functools import cache
from typing import Any

from pydantic_core import PydanticCustomError

from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalInteger,
    CanonicalRational,
)
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
    IntegerPolynomialCompositionResult,
    IntegerPolynomialContentResult,
    IntegerPolynomialEvaluationResult,
    IntegerPolynomialGcdResult,
    IntegerPolynomialPrimitivePartResult,
    IntegerPolynomialShiftResult,
    RationalPartialFractionResult,
    RationalPartialFractionTerm,
    RationalPolynomialDerivativeResult,
    RationalPolynomialDivisionResult,
    RationalPolynomialEvaluationResult,
    RationalPolynomialIntegralResult,
    _validation_error,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
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


def _admit_integer_pair(left: IntegerPolynomial, right: IntegerPolynomial) -> None:
    _admit_integer(left)
    _admit_integer(right)


def _admit_integer_evaluation(
    polynomial: IntegerPolynomial, point: CanonicalInteger
) -> None:
    _admit_integer(polynomial)
    if len(point.lstrip("-")) > _MAX_INTEGER_COEFFICIENT_DIGITS:
        raise _validation_error("evaluation point exceeds the decimal-digit budget")


def _admit_integer_composition(
    outer: IntegerPolynomial, inner: IntegerPolynomial
) -> None:
    _admit_integer(outer)
    _admit_integer(inner)
    if (len(outer.coefficients) - 1) * (
        len(inner.coefficients) - 1
    ) > _MAX_ELEMENTARY_DEGREE:
        raise _validation_error(
            f"composition exceeds the degree-{_MAX_ELEMENTARY_DEGREE} output budget"
        )


def _admit_rational(polynomial: RationalPolynomial) -> None:
    if len(polynomial.variables) != 1:
        raise _validation_error("elementary polynomial operations require one variable")
    require_polynomial_budget(
        polynomial,
        maximum_terms=_MAX_GCD_TERMS,
        maximum_exponent=_MAX_ELEMENTARY_DEGREE,
    )


def _admit_rational_evaluation(
    polynomial: RationalPolynomial,
    point: CanonicalRational,
) -> None:
    _admit_rational(polynomial)
    numerator_digits, denominator_digits = rational_evaluation_component_digit_bounds(
        polynomial,
        (point,),
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


def _admit_division(left: RationalPolynomial, right: RationalPolynomial) -> None:
    if left.variables != right.variables:
        raise _validation_error("polynomials must use the same ordered variables")
    if len(left.variables) != 1:
        raise _validation_error("polynomial division requires one variable")
    if not right.polynomial.terms:
        raise _validation_error("divisor polynomial must be nonzero")
    for polynomial in (left, right):
        require_polynomial_budget(
            polynomial,
            maximum_terms=_MAX_GCD_TERMS,
            maximum_exponent=_MAX_ELEMENTARY_DEGREE,
        )


def _admit_partial_fractions(
    numerator: RationalPolynomial, denominator: RationalPolynomial
) -> None:
    if numerator.variables != denominator.variables:
        raise _validation_error("numerator and denominator must use the same ring")
    if len(numerator.variables) != 1:
        raise _validation_error("partial fractions require one variable")
    if not denominator.polynomial.terms:
        raise _validation_error("denominator polynomial must be nonzero")
    for polynomial in (numerator, denominator):
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
    left: IntegerPolynomial, right: IntegerPolynomial
) -> IntegerPolynomialGcdResult:
    """Compute the exact GCD and contents of two canonical integer polynomials."""

    _run_admission(lambda: _admit_integer_pair(left, right))
    left_backend = _integer_poly(left)
    right_backend = _integer_poly(right)
    gcd = left_backend.gcd(right_backend)
    return IntegerPolynomialGcdResult(
        gcd=_integer_wire(gcd),
        left_content=format_canonical_integer(int(left_backend.content())),
        right_content=format_canonical_integer(int(right_backend.content())),
        gcd_content=format_canonical_integer(int(gcd.content())),
    )


def integer_polynomial_content(
    polynomial: IntegerPolynomial,
) -> IntegerPolynomialContentResult:
    """Return the nonnegative coefficient content of a canonical polynomial."""

    _run_admission(lambda: _admit_integer(polynomial))
    return IntegerPolynomialContentResult(
        content=format_canonical_integer(int(_integer_poly(polynomial).content()))
    )


def integer_polynomial_primitive_part(
    polynomial: IntegerPolynomial,
) -> IntegerPolynomialPrimitivePartResult:
    """Return content, primitive part, and exact reconstruction."""

    _run_admission(lambda: _admit_integer(polynomial))
    source = _integer_poly(polynomial)
    content, primitive = source.primitive()
    reconstructed = primitive.mul_ground(content)
    return IntegerPolynomialPrimitivePartResult(
        content=format_canonical_integer(int(content)),
        primitive_part=_integer_wire(primitive),
        reconstruction=_integer_wire(reconstructed),
    )


def integer_polynomial_evaluate(
    polynomial: IntegerPolynomial, point: CanonicalInteger
) -> IntegerPolynomialEvaluationResult:
    """Evaluate a canonical integer polynomial at one integer point."""

    _run_admission(lambda: _admit_integer_evaluation(polynomial, point))
    parsed_point = parse_canonical_integer(point)
    value = _integer_poly(polynomial).eval(parsed_point)
    return IntegerPolynomialEvaluationResult(
        point=point,
        value=format_canonical_integer(int(value)),
    )


def integer_polynomial_compose(
    outer: IntegerPolynomial, inner: IntegerPolynomial
) -> IntegerPolynomialCompositionResult:
    """Compose two canonical integer polynomials."""

    _run_admission(lambda: _admit_integer_composition(outer, inner))
    composition = _integer_poly(outer).compose(_integer_poly(inner))
    return IntegerPolynomialCompositionResult(composition=_integer_wire(composition))


def integer_polynomial_shift(
    polynomial: IntegerPolynomial, shift: int
) -> IntegerPolynomialShiftResult:
    """Compute ``p(x + a)`` using SymPy's exact dense shift."""
    _run_admission(lambda: _admit_integer(polynomial))
    shifted = _integer_poly(polynomial).shift(shift)
    return IntegerPolynomialShiftResult(
        shift=shift,
        shifted=_integer_wire(shifted),
    )


def rational_polynomial_division(
    left: RationalPolynomial, right: RationalPolynomial
) -> RationalPolynomialDivisionResult:
    """Divide two canonical univariate rational polynomials exactly."""

    _run_admission(lambda: _admit_division(left, right))
    left_backend = rational_polynomial_to_sympy(left)
    right_backend = rational_polynomial_to_sympy(right)
    quotient, remainder, reconstruction = polynomials.divide(
        left_backend, right_backend
    )
    variables = left.variables
    return RationalPolynomialDivisionResult(
        quotient=rational_polynomial_from_sympy(quotient, variables),
        remainder=rational_polynomial_from_sympy(remainder, variables),
        reconstruction=rational_polynomial_from_sympy(reconstruction, variables),
    )


def rational_polynomial_evaluate(
    polynomial: RationalPolynomial, point: CanonicalRational
) -> RationalPolynomialEvaluationResult:
    """Evaluate one canonical rational polynomial at an exact rational point."""

    _run_admission(lambda: _admit_rational_evaluation(polynomial, point))
    point_value = point.as_fraction()
    from sympy import Rational

    value = polynomials.evaluate(
        rational_polynomial_to_sympy(polynomial),
        Rational(point_value.numerator, point_value.denominator),
    )
    return RationalPolynomialEvaluationResult(
        point=point,
        value=rational_from_sympy(value),
    )


def rational_polynomial_derivative(
    polynomial: RationalPolynomial,
) -> RationalPolynomialDerivativeResult:
    """Return the exact derivative of a canonical rational polynomial."""

    _run_admission(lambda: _admit_rational(polynomial))
    return RationalPolynomialDerivativeResult(
        derivative=rational_polynomial_from_sympy(
            polynomials.derivative(rational_polynomial_to_sympy(polynomial)),
            polynomial.variables,
        )
    )


def rational_polynomial_integral(
    polynomial: RationalPolynomial,
) -> RationalPolynomialIntegralResult:
    """Return the exact zero-constant-term integral."""

    _run_admission(lambda: _admit_rational(polynomial))
    return RationalPolynomialIntegralResult(
        antiderivative=rational_polynomial_from_sympy(
            polynomials.integral(rational_polynomial_to_sympy(polynomial)),
            polynomial.variables,
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
    numerator: RationalPolynomial, denominator: RationalPolynomial
) -> RationalPartialFractionResult:
    """Return the exact partial-fraction decomposition of a rational function."""

    _run_admission(lambda: _admit_partial_fractions(numerator, denominator))
    from sympy import Add, Poly, cancel, fraction, together

    variables = numerator.variables
    numerator_polynomial = rational_polynomial_to_sympy(numerator)
    denominator_polynomial = rational_polynomial_to_sympy(denominator)
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
