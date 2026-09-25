from fractions import Fraction
from math import factorial

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.ore_algebras._models import ShiftOreOperator, ShiftOreTerm
from jacobian.math.ore_algebras.proper_hypergeometric_actions import (
    proper_hypergeometric_operator_action,
)
from jacobian.math.ore_algebras.proper_hypergeometric_terms import (
    IntegerAffineFactorial,
    ProperHypergeometricTerm,
)
from jacobian.math.polynomials.values import (
    RationalFunction,
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _polynomial(terms):
    return RationalPolynomial(
        variables=("n", "k"),
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=coefficient, den=1),
                    exponents=exponents,
                )
                for exponents, coefficient in terms
            )
        ),
    )


def _binomial_term():
    return ProperHypergeometricTerm(
        polynomial=_polynomial([((0, 0), 1)]),
        factorial_factors=(
            IntegerAffineFactorial(
                n_coefficient=0, k_coefficient=1, offset=0, power=-1
            ),
            IntegerAffineFactorial(
                n_coefficient=1, k_coefficient=-1, offset=0, power=-1
            ),
            IntegerAffineFactorial(n_coefficient=1, k_coefficient=0, offset=0, power=1),
        ),
    )


def _rf_value(value: RationalFunction, n: int, k: int) -> Fraction:
    def evaluate(polynomial):
        return sum(
            term.coefficient.as_fraction()
            * n ** term.exponents[0]
            * k ** term.exponents[1]
            for term in polynomial.terms
        )

    return evaluate(value.numerator) / evaluate(value.denominator)


def _bivariate_rf(numerator_terms, denominator_terms):
    def polynomial(terms):
        return SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=coefficient, den=1),
                    exponents=exponents,
                )
                for exponents, coefficient in terms
            )
        )

    return RationalFunction(
        variables=("n", "k"),
        numerator=polynomial(numerator_terms),
        denominator=polynomial(denominator_terms),
    )


def _direct_binomial(n: int, k: int) -> int:
    return factorial(n) // (factorial(k) * factorial(n - k))


def _operator(*terms):
    one_poly = SparseRationalPolynomial(
        terms=(
            RationalPolynomialTerm(
                coefficient=CanonicalRational(num=1, den=1), exponents=(0,)
            ),
        )
    )
    one = RationalFunction(variables=("n",), numerator=one_poly, denominator=one_poly)
    return ShiftOreOperator(
        terms=tuple(
            ShiftOreTerm(exponent=exponent, coefficient=one) for exponent in terms
        )
    )


def test_shift_operator_action_matches_independent_binomial_values():
    term = _binomial_term()
    result = proper_hypergeometric_operator_action(_operator(1), term)
    assert result.relative_multiplier == _bivariate_rf(
        [((1, 0), 1), ((0, 0), 1)],
        [((1, 0), 1), ((0, 1), -1), ((0, 0), 1)],
    )
    for n, k in ((5, 2), (8, 3), (12, 7)):
        multiplier = _rf_value(result.relative_multiplier, n, k)
        assert multiplier * _direct_binomial(n, k) == _direct_binomial(n + 1, k)
    assert _rf_value(result.relative_multiplier, 5, 2) == Fraction(3, 2)


def test_multiple_operator_terms_sum_their_exact_shift_actions():
    term = _binomial_term()
    result = proper_hypergeometric_operator_action(_operator(0, 1), term)
    assert result.relative_multiplier == _bivariate_rf(
        [((1, 0), 2), ((0, 1), -1), ((0, 0), 2)],
        [((1, 0), 1), ((0, 1), -1), ((0, 0), 1)],
    )
    assert _rf_value(result.relative_multiplier, 5, 2) == Fraction(5, 2)
    # Directly evaluate (I + S)T at an integer point, independent of quotient formulas.
    assert _rf_value(result.relative_multiplier, 5, 2) * _direct_binomial(
        5, 2
    ) == _direct_binomial(5, 2) + _direct_binomial(6, 2)


def test_left_coefficient_remains_unshifted_in_ore_action():
    n_coefficient = RationalFunction(
        variables=("n",),
        numerator=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=1, den=1), exponents=(1,)
                ),
            )
        ),
        denominator=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=1, den=1), exponents=(0,)
                ),
            )
        ),
    )
    operator = ShiftOreOperator(
        terms=(ShiftOreTerm(exponent=1, coefficient=n_coefficient),)
    )
    result = proper_hypergeometric_operator_action(operator, _binomial_term())
    # n*S acts as n*T(n+1,k), so its relative multiplier is n times
    # (n+1)/(n+1-k), not (n+1) times that quotient.
    assert _rf_value(result.relative_multiplier, 5, 2) == Fraction(15, 2)
    assert _rf_value(result.relative_multiplier, 5, 2) * _direct_binomial(
        5, 2
    ) == 5 * _direct_binomial(6, 2)


def test_zero_operator_and_zero_term_have_canonical_zero_multiplier():
    zero_operator = ShiftOreOperator(terms=())
    result = proper_hypergeometric_operator_action(zero_operator, _binomial_term())
    assert result.relative_multiplier.numerator.terms == ()

    zero_term = ProperHypergeometricTerm(polynomial=_polynomial([]))
    result = proper_hypergeometric_operator_action(_operator(0, 2), zero_term)
    assert result.relative_multiplier.numerator.terms == ()


def test_large_expansion_is_rejected_before_rational_normalization():
    with pytest.raises(OperationResourceAdmissionError, match="256-term"):
        proper_hypergeometric_operator_action(_operator(9), _binomial_term())


def test_native_boundary_rejects_noncanonical_term_values_with_owner_error():
    with pytest.raises(OperationDomainValidationError, match="canonical proper"):
        proper_hypergeometric_operator_action(_operator(0), None)
