from collections.abc import Iterable
from fractions import Fraction
from math import factorial
from typing import cast

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


def _polynomial(
    terms: Iterable[tuple[tuple[int, int], int]],
) -> RationalPolynomial:
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


def _binomial_term() -> ProperHypergeometricTerm:
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
    def evaluate(polynomial: SparseRationalPolynomial) -> Fraction:
        return sum(
            (
                term.coefficient.as_fraction()
                * n ** term.exponents[0]
                * k ** term.exponents[1]
                for term in polynomial.terms
            ),
            Fraction(0),
        )

    return evaluate(value.numerator) / evaluate(value.denominator)


def _bivariate_rf(
    numerator_terms: Iterable[tuple[tuple[int, int], int]],
    denominator_terms: Iterable[tuple[tuple[int, int], int]],
) -> RationalFunction:
    def polynomial(
        terms: Iterable[tuple[tuple[int, int], int]],
    ) -> SparseRationalPolynomial:
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


def _operator(*terms: int) -> ShiftOreOperator:
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


def test_shift_operator_action_matches_independent_binomial_values() -> None:
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


def test_multiple_operator_terms_sum_their_exact_shift_actions() -> None:
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


def test_left_coefficient_remains_unshifted_in_ore_action() -> None:
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


def test_zero_operator_and_zero_term_have_canonical_zero_multiplier() -> None:
    zero_operator = ShiftOreOperator(terms=())
    result = proper_hypergeometric_operator_action(zero_operator, _binomial_term())
    assert result.relative_multiplier.numerator.terms == ()

    zero_term = ProperHypergeometricTerm(polynomial=_polynomial([]))
    result = proper_hypergeometric_operator_action(_operator(0, 2), zero_term)
    assert result.relative_multiplier.numerator.terms == ()


def test_degree_nine_expansion_is_accepted_and_larger_expansion_is_rejected() -> None:
    result = proper_hypergeometric_operator_action(_operator(9), _binomial_term())
    assert _rf_value(result.relative_multiplier, 20, 4) * _direct_binomial(
        20, 4
    ) == _direct_binomial(29, 4)

    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        proper_hypergeometric_operator_action(_operator(16), _binomial_term())
    assert (
        exc_info.value.errors()[0]["type"]
        == "ore_algebra.hypergeometric_action_term_budget"
    )


def test_sparse_univariate_shift_product_uses_its_actual_axis_support() -> None:
    term = ProperHypergeometricTerm(
        polynomial=_polynomial([((0, 0), 1)]),
        factorial_factors=(
            IntegerAffineFactorial(
                n_coefficient=1, k_coefficient=0, offset=0, power=12
            ),
        ),
    )
    result = proper_hypergeometric_operator_action(_operator(2), term)
    assert _rf_value(result.relative_multiplier, 3, 7) == 4**12 * 5**12


def test_tenth_shift_of_twelfth_factorial_power_fits_admitted_bounds() -> None:
    term = ProperHypergeometricTerm(
        polynomial=_polynomial([((0, 0), 1)]),
        factorial_factors=(
            IntegerAffineFactorial(
                n_coefficient=1, k_coefficient=0, offset=0, power=12
            ),
        ),
    )

    result = proper_hypergeometric_operator_action(_operator(10), term)

    assert len(result.relative_multiplier.numerator.terms) == 121
    assert len(result.relative_multiplier.denominator.terms) == 1
    assert (
        _rf_value(result.relative_multiplier, 3, 7)
        == (factorial(13) // factorial(3)) ** 12
    )


def test_n_action_does_not_require_admission_of_the_k_quotient() -> None:
    term = ProperHypergeometricTerm(
        polynomial=_polynomial([((0, 0), 1)]),
        factorial_factors=(
            IntegerAffineFactorial(
                n_coefficient=0, k_coefficient=128, offset=0, power=32
            ),
        ),
    )
    result = proper_hypergeometric_operator_action(_operator(1), term)
    assert _rf_value(result.relative_multiplier, 3, 2) == 1


def test_native_boundary_rejects_noncanonical_term_values_with_owner_error() -> None:
    with pytest.raises(OperationDomainValidationError) as exc_info:
        proper_hypergeometric_operator_action(
            _operator(0), cast(ProperHypergeometricTerm, None)
        )
    assert (
        exc_info.value.errors()[0]["type"] == "ore_algebra.proper_hypergeometric_term"
    )
