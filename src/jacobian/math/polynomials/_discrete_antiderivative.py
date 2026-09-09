"""Selected-variable rational discrete antiderivatives."""

from fractions import Fraction
from math import comb

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_EXPONENT,
    MAX_POLYNOMIAL_TERMS,
    PolynomialVariable,
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


class RationalDiscreteAntiderivativeRequest(StrictModel):
    polynomial: RationalPolynomial
    variable: PolynomialVariable


class RationalDiscreteAntiderivativeResult(StrictModel):
    source: RationalDiscreteAntiderivativeRequest
    antiderivative: RationalPolynomial
    reconstructed_difference: RationalPolynomial


def _polynomial(
    variables: tuple[str, ...], coefficients: dict[tuple[int, ...], Fraction]
) -> RationalPolynomial:
    return RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_fraction(coefficient),
                    exponents=exponents,
                )
                for exponents, coefficient in sorted(coefficients.items(), reverse=True)
                if coefficient
            )
        ),
    )


def rational_discrete_antiderivative(
    request: RationalDiscreteAntiderivativeRequest,
) -> RationalDiscreteAntiderivativeResult:
    source = request.polynomial
    if request.variable not in source.variables:
        raise OperationDomainValidationError(
            location=("variable",),
            code="polynomial.discrete_antiderivative.variable_axis",
            message="selected variable must belong to the polynomial axis",
        )
    variable_index = source.variables.index(request.variable)
    output_bound = sum(
        term.exponents[variable_index] + 1 for term in source.polynomial.terms
    )
    maximum_degree = max(
        (term.exponents[variable_index] for term in source.polynomial.terms),
        default=0,
    )
    if output_bound > MAX_POLYNOMIAL_TERMS or maximum_degree >= MAX_POLYNOMIAL_EXPONENT:
        raise OperationResourceAdmissionError(
            location=("polynomial",),
            code="polynomial.discrete_antiderivative.output_bound",
            message="discrete antiderivative exceeds the sparse support or exponent bound",
        )
    groups: dict[tuple[int, ...], dict[int, Fraction]] = {}
    for term in source.polynomial.terms:
        other = (
            *term.exponents[:variable_index],
            *term.exponents[variable_index + 1 :],
        )
        groups.setdefault(other, {})[term.exponents[variable_index]] = (
            term.coefficient.as_fraction()
        )
    answer: dict[tuple[int, ...], Fraction] = {}
    for other, coefficients in groups.items():
        residual = dict(coefficients)
        for degree in range(max(residual, default=-1), -1, -1):
            leading = residual.get(degree, Fraction())
            if not leading:
                continue
            antiderivative_coefficient = leading / (degree + 1)
            exponent = (
                *other[:variable_index],
                degree + 1,
                *other[variable_index:],
            )
            answer[exponent] = antiderivative_coefficient
            for lower_degree in range(degree + 1):
                residual[lower_degree] = residual.get(
                    lower_degree, Fraction()
                ) - antiderivative_coefficient * comb(degree + 1, lower_degree)
    reconstructed: dict[tuple[int, ...], Fraction] = {}
    for exponents, coefficient in answer.items():
        degree = exponents[variable_index]
        for lower_degree in range(degree):
            target = list(exponents)
            target[variable_index] = lower_degree
            key = tuple(target)
            reconstructed[key] = reconstructed.get(
                key, Fraction()
            ) + coefficient * comb(degree, lower_degree)
    antiderivative = _polynomial(source.variables, answer)
    difference = _polynomial(source.variables, reconstructed)
    if difference != source:
        raise RuntimeError("discrete antiderivative reconstruction failed")
    return RationalDiscreteAntiderivativeResult(
        source=request,
        antiderivative=antiderivative,
        reconstructed_difference=difference,
    )


__all__ = ["rational_discrete_antiderivative"]
