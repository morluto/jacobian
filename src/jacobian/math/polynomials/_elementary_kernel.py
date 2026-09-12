"""Exact elementary polynomial operations backed by SymPy ``Poly`` APIs."""

from __future__ import annotations

from functools import cache
from math import gcd
from typing import Any, Literal

from pydantic_core import PydanticCustomError

from jacobian._exact import (
    MAX_CANONICAL_INTEGER_DIGITS,
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalRational,
)
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
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
    MAX_POLYNOMIAL_TERMS,
    RationalPolynomial,
    rational_evaluation_component_digit_bounds,
    require_polynomial_budget,
)

_CANONICAL_INTEGER_LIMIT = 10**MAX_CANONICAL_INTEGER_DIGITS


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
        abs(coefficient) >= 10**_MAX_INTEGER_COEFFICIENT_DIGITS
        for coefficient in polynomial.coefficients
    ):
        raise _validation_error("integer coefficient exceeds the decimal-digit budget")


def _admit_primitive_part(polynomial: IntegerPolynomial) -> None:
    """Admit content/primitive decomposition on the integer-polynomial carrier."""

    coefficients = polynomial.coefficients
    if not isinstance(coefficients, tuple) or not coefficients:
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="polynomial.primitive_part_empty",
            message="a canonical integer polynomial has at least one coefficient",
        )
    if any(type(coefficient) is not int for coefficient in coefficients):
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="polynomial.primitive_part_coefficients",
            message="primitive-part coefficients must be exact integers",
        )
    if len(coefficients) > 1 and coefficients[0] == 0:
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="polynomial.primitive_part_shape",
            message="a canonical integer polynomial omits leading zeros",
        )
    if len(coefficients) > MAX_POLYNOMIAL_TERMS:
        raise OperationResourceAdmissionError(
            location=("polynomial",),
            code="polynomial.primitive_part_term_bound",
            message="primitive decomposition exceeds the integer-polynomial carrier",
        )
    if any(
        abs(coefficient) >= _CANONICAL_INTEGER_LIMIT
        for coefficient in coefficients
        if abs(coefficient).bit_length() > MAX_CANONICAL_INTEGER_DIGITS
    ):
        raise OperationResourceAdmissionError(
            location=("polynomial",),
            code="polynomial.primitive_part_integer_digits",
            message="a coefficient exceeds the canonical integer representation envelope",
        )


def _admit_integer_pair(left: IntegerPolynomial, right: IntegerPolynomial) -> None:
    _admit_integer(left)
    _admit_integer(right)


def _admit_integer_evaluation(polynomial: IntegerPolynomial, point: int) -> None:
    _admit_integer(polynomial)
    if type(point) is not int or abs(point) >= 10**_MAX_INTEGER_COEFFICIENT_DIGITS:
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
        list(polynomial.coefficients),
        _x(),
        domain="ZZ",
    )


def _integer_value(polynomial: Any) -> IntegerPolynomial:
    return IntegerPolynomial(
        coefficients=tuple(int(coefficient) for coefficient in polynomial.all_coeffs())
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
        gcd=_integer_value(gcd),
        left_content=int(left_backend.content()),
        right_content=int(right_backend.content()),
        gcd_content=int(gcd.content()),
    )


def integer_polynomial_content(
    polynomial: IntegerPolynomial,
) -> IntegerPolynomialContentResult:
    """Return the nonnegative coefficient content of a canonical polynomial."""

    _run_admission(lambda: _admit_integer(polynomial))
    return IntegerPolynomialContentResult(
        content=int(_integer_poly(polynomial).content())
    )


MAX_PRIMITIVE_PART_RESULT_DIGITS = 8_000_000


def _retained_integer_digits(value: int) -> int:
    if value == 0:
        return 1
    return (abs(value).bit_length() * 30103) // 100000 + 1


def integer_polynomial_primitive_part(
    polynomial: IntegerPolynomial,
) -> IntegerPolynomialPrimitivePartResult:
    """Return sign, content, positive-leading primitive part, and reconstruction."""

    _run_admission(lambda: _admit_primitive_part(polynomial))
    coefficients = polynomial.coefficients
    if not any(coefficients):
        zero = IntegerPolynomial(coefficients=(0,))
        return IntegerPolynomialPrimitivePartResult._from_kernel(
            sign=1,
            content=0,
            primitive_part=zero,
            degree=0,
            reconstruction=zero,
        )
    source_digits = sum(
        _retained_integer_digits(coefficient) for coefficient in coefficients
    )
    if 2 * source_digits + 1 > MAX_PRIMITIVE_PART_RESULT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("polynomial",),
            code="polynomial.content_profile_result_digits",
            message="the retained profile coefficients exceed the exact output bound",
        )
    sign: Literal[-1, 1] = 1 if coefficients[0] > 0 else -1
    content = 0
    for coefficient in coefficients:
        content = gcd(content, abs(coefficient))
    primitive = tuple((sign * coefficient) // content for coefficient in coefficients)
    reconstruction = tuple(sign * content * value for value in primitive)
    return IntegerPolynomialPrimitivePartResult._from_kernel(
        sign=sign,
        content=content,
        primitive_part=IntegerPolynomial(coefficients=primitive),
        degree=len(primitive) - 1,
        reconstruction=IntegerPolynomial(coefficients=reconstruction),
    )


def integer_polynomial_evaluate(
    polynomial: IntegerPolynomial, point: int
) -> IntegerPolynomialEvaluationResult:
    """Evaluate a canonical integer polynomial at one integer point."""

    _run_admission(lambda: _admit_integer_evaluation(polynomial, point))
    value = _integer_poly(polynomial).eval(point)
    return IntegerPolynomialEvaluationResult(
        point=point,
        value=int(value),
    )


def integer_polynomial_compose(
    outer: IntegerPolynomial, inner: IntegerPolynomial
) -> IntegerPolynomialCompositionResult:
    """Compose two canonical integer polynomials."""

    _run_admission(lambda: _admit_integer_composition(outer, inner))
    composition = _integer_poly(outer).compose(_integer_poly(inner))
    return IntegerPolynomialCompositionResult(composition=_integer_value(composition))


def integer_polynomial_shift(
    polynomial: IntegerPolynomial, shift: int
) -> IntegerPolynomialShiftResult:
    """Compute ``p(x + a)`` using SymPy's exact dense shift."""
    _run_admission(lambda: _admit_integer(polynomial))
    shifted = _integer_poly(polynomial).shift(shift)
    return IntegerPolynomialShiftResult(
        shift=shift,
        shifted=_integer_value(shifted),
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
