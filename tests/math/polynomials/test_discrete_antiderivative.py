"""Selected-variable rational discrete antiderivative tests."""

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.polynomials._discrete_antiderivative import (
    RationalDiscreteAntiderivativeRequest,
    rational_discrete_antiderivative,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _polynomial(terms: tuple[tuple[int, tuple[int, ...]], ...]) -> RationalPolynomial:
    return RationalPolynomial(
        variables=("k", "N"),
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=coefficient, den=1),
                    exponents=exponents,
                )
                for coefficient, exponents in terms
            )
        ),
    )


def test_retained_k_squared_n_minus_k_squared_fixture() -> None:
    source = _polynomial(((1, (4, 0)), (-2, (3, 1)), (1, (2, 2))))
    result = rational_discrete_antiderivative(
        RationalDiscreteAntiderivativeRequest(polynomial=source, variable="k")
    )
    assert result.reconstructed_difference == source
    assert all(
        term.exponents[0] >= 1 for term in result.antiderivative.polynomial.terms
    )


def test_other_variable_is_a_coefficient_parameter() -> None:
    source = _polynomial(((1, (1, 1)),))
    result = rational_discrete_antiderivative(
        RationalDiscreteAntiderivativeRequest(polynomial=source, variable="k")
    )
    coefficients = {
        term.exponents: term.coefficient.as_fraction()
        for term in result.antiderivative.polynomial.terms
    }
    assert coefficients == {(2, 1): Fraction(1, 2), (1, 1): Fraction(-1, 2)}


def test_denominator_growth_is_rejected_before_result_construction() -> None:
    denominator = 5 * 10**32_767 + 1
    source = RationalPolynomial(
        variables=("k",),
        polynomial=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=1, den=denominator),
                    exponents=(1,),
                ),
            )
        ),
    )
    with pytest.raises(OperationResourceAdmissionError):
        rational_discrete_antiderivative(
            RationalDiscreteAntiderivativeRequest(polynomial=source, variable="k")
        )
