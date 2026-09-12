"""Exact sparse rational Laurent-polynomial multiplication."""

from fractions import Fraction

from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalRational,
    canonical_rational_component_digits,
)
from jacobian._execution import request_checkpoint
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_EXPONENT,
    MAX_POLYNOMIAL_TERMS,
    RationalLaurentPolynomial,
    RationalLaurentPolynomialTerm,
)


class RationalLaurentMultiplyRequest(StrictModel):
    left: RationalLaurentPolynomial
    right: RationalLaurentPolynomial


def _maximum_coefficient_digits(
    left: RationalLaurentPolynomial, right: RationalLaurentPolynomial
) -> int:
    """Bound one collected coefficient before exact convolution begins."""

    collisions = min(len(left.terms), len(right.terms))
    left_digits = max(
        (canonical_rational_component_digits(term.coefficient) for term in left.terms),
        default=1,
    )
    right_digits = max(
        (canonical_rational_component_digits(term.coefficient) for term in right.terms),
        default=1,
    )
    addition_digits = len(str(collisions)) if collisions > 1 else 0
    return collisions * (left_digits + right_digits) + addition_digits


def _result_from_coefficients(
    variables: tuple[str, ...], coefficients: dict[tuple[int, ...], Fraction]
) -> RationalLaurentPolynomial:
    terms: list[RationalLaurentPolynomialTerm] = []
    for exponents, value in sorted(coefficients.items(), reverse=True):
        if not value:
            continue
        component_digits = max(
            len(format_canonical_integer(abs(value.numerator))),
            len(format_canonical_integer(value.denominator)),
        )
        if component_digits > MAX_CANONICAL_RATIONAL_DIGITS:
            raise OperationResourceAdmissionError(
                location=("right", "terms"),
                code="polynomial.laurent.output_growth",
                message="a reduced Laurent coefficient exceeds the exact-output bound",
            )
        terms.append(
            RationalLaurentPolynomialTerm(
                coefficient=CanonicalRational(
                    num=value.numerator, den=value.denominator
                ),
                exponents=exponents,
            )
        )
    return RationalLaurentPolynomial(variables=variables, terms=tuple(terms))


def rational_laurent_multiply(
    left: RationalLaurentPolynomial, right: RationalLaurentPolynomial
) -> RationalLaurentPolynomial:
    if left.variables != right.variables:
        raise OperationDomainValidationError(
            location=("right", "variables"),
            code="polynomial.laurent.axis_mismatch",
            message="Laurent factors must use the same ordered variable axis",
        )
    if not left.terms or not right.terms:
        return RationalLaurentPolynomial(variables=left.variables, terms=())
    request_checkpoint("before Laurent semantic admission")
    work = len(left.terms) * len(right.terms)
    if work > MAX_POLYNOMIAL_TERMS:
        raise OperationResourceAdmissionError(
            location=("right", "terms"),
            code="polynomial.laurent.convolution_bound",
            message=f"sparse convolution requires {work} term products; maximum is {MAX_POLYNOMIAL_TERMS}",
        )
    coefficient_digits = _maximum_coefficient_digits(left, right)
    if coefficient_digits > MAX_CANONICAL_RATIONAL_DIGITS:
        raise OperationResourceAdmissionError(
            location=("right", "terms"),
            code="polynomial.laurent.coefficient_growth",
            message="Laurent convolution coefficients exceed the exact-output bound",
        )
    request_checkpoint("after Laurent semantic admission")

    # A monomial only shifts the other support and scales its coefficients;
    # avoid paying for a general pair convolution in that common case.
    if len(left.terms) == 1 or len(right.terms) == 1:
        monomial, source = (
            (left.terms[0], right) if len(left.terms) == 1 else (right.terms[0], left)
        )
        factor = monomial.coefficient.as_fraction()
        monomial_coefficients: dict[tuple[int, ...], Fraction] = {}
        for term in source.terms:
            exponents = tuple(
                a + b for a, b in zip(monomial.exponents, term.exponents, strict=True)
            )
            if any(abs(exponent) > MAX_POLYNOMIAL_EXPONENT for exponent in exponents):
                raise OperationResourceAdmissionError(
                    location=("right", "terms"),
                    code="polynomial.laurent.exponent_growth",
                    message="product exponent exceeds the Laurent representation bound",
                )
            monomial_coefficients[exponents] = factor * term.coefficient.as_fraction()
        result = _result_from_coefficients(left.variables, monomial_coefficients)
        request_checkpoint("after Laurent monomial result construction")
        return result

    coefficients: dict[tuple[int, ...], Fraction] = {}
    pairs = 0
    for left_term in left.terms:
        for right_term in right.terms:
            if pairs % 128 == 0:
                request_checkpoint("during Laurent convolution")
            exponents = tuple(
                a + b
                for a, b in zip(left_term.exponents, right_term.exponents, strict=True)
            )
            if any(abs(exponent) > MAX_POLYNOMIAL_EXPONENT for exponent in exponents):
                raise OperationResourceAdmissionError(
                    location=("right", "terms"),
                    code="polynomial.laurent.exponent_growth",
                    message="product exponent exceeds the Laurent representation bound",
                )
            coefficients[exponents] = coefficients.get(exponents, Fraction()) + (
                left_term.coefficient.as_fraction()
                * right_term.coefficient.as_fraction()
            )
            pairs += 1
    result = _result_from_coefficients(left.variables, coefficients)
    request_checkpoint("after Laurent result construction")
    return result


def _run(request: RationalLaurentMultiplyRequest) -> RationalLaurentPolynomial:
    return rational_laurent_multiply(request.left, request.right)
