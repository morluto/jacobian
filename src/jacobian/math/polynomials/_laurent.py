"""Exact sparse rational Laurent-polynomial multiplication."""

from fractions import Fraction

from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalRational,
    canonical_rational_component_digits,
)
from jacobian._models import StrictModel
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
    work = len(left.terms) * len(right.terms)
    if work > MAX_POLYNOMIAL_TERMS:
        raise OperationResourceAdmissionError(
            location=("right", "terms"),
            code="polynomial.laurent.convolution_bound",
            message=f"sparse convolution requires {work} term products; maximum is {MAX_POLYNOMIAL_TERMS}",
        )
    # Check the full exponent envelope before entering the convolution.  This
    # keeps a mathematically invalid product from doing partial exact work and
    # makes the admission decision independent of cancellation in the result.
    max_left = tuple(
        max(abs(term.exponents[index]) for term in left.terms)
        for index in range(len(left.variables))
    )
    max_right = tuple(
        max(abs(term.exponents[index]) for term in right.terms)
        for index in range(len(right.variables))
    )
    if any(
        left_bound + right_bound > MAX_POLYNOMIAL_EXPONENT
        for left_bound, right_bound in zip(max_left, max_right, strict=True)
    ):
        raise OperationResourceAdmissionError(
            location=("right", "terms"),
            code="polynomial.laurent.exponent_growth",
            message="product exponent may exceed the Laurent representation bound",
        )
    coefficient_digits = work * (
        max(
            canonical_rational_component_digits(term.coefficient) for term in left.terms
        )
        + max(
            canonical_rational_component_digits(term.coefficient)
            for term in right.terms
        )
    ) + len(str(work))
    if coefficient_digits > MAX_CANONICAL_RATIONAL_DIGITS:
        raise OperationResourceAdmissionError(
            location=("right", "terms"),
            code="polynomial.laurent.coefficient_growth",
            message="Laurent convolution coefficients exceed the exact-output bound",
        )
    coefficients: dict[tuple[int, ...], Fraction] = {}
    for left_term in left.terms:
        for right_term in right.terms:
            exponents = tuple(
                a + b
                for a, b in zip(left_term.exponents, right_term.exponents, strict=True)
            )
            coefficients[exponents] = coefficients.get(exponents, Fraction()) + (
                left_term.coefficient.as_fraction()
                * right_term.coefficient.as_fraction()
            )
    terms = tuple(
        RationalLaurentPolynomialTerm(
            coefficient=CanonicalRational(num=value.numerator, den=value.denominator),
            exponents=exponents,
        )
        for exponents, value in sorted(coefficients.items(), reverse=True)
        if value
    )
    return RationalLaurentPolynomial(variables=left.variables, terms=terms)


def _run(request: RationalLaurentMultiplyRequest) -> RationalLaurentPolynomial:
    return rational_laurent_multiply(request.left, request.right)
