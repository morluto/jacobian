from fractions import Fraction
from math import factorial

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.ore_algebras.proper_hypergeometric_terms import (
    IntegerAffineFactorial,
    ProperHypergeometricTerm,
    proper_hypergeometric_shift_quotients,
)
from jacobian.math.polynomials.values import (
    RationalFunction,
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def polynomial(terms):
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
        polynomial=polynomial([((0, 0), 1)]),
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


def _eval_rf(value: RationalFunction, n: int, k: int) -> Fraction:
    def eval_poly(poly):
        return sum(
            term.coefficient.as_fraction()
            * n ** term.exponents[0]
            * k ** term.exponents[1]
            for term in poly.terms
        )

    return eval_poly(value.numerator) / eval_poly(value.denominator)


def _direct(term: ProperHypergeometricTerm, n: int, k: int) -> Fraction:
    total = Fraction(0)
    for item in term.polynomial.polynomial.terms:
        total += (
            item.coefficient.as_fraction()
            * n ** item.exponents[0]
            * k ** item.exponents[1]
        )
    for factor in term.factorial_factors:
        argument = factor.n_coefficient * n + factor.k_coefficient * k + factor.offset
        if argument < 0:
            if factor.power < 0:
                return Fraction(0)
            raise ValueError("point is outside factorial domain")
        total *= Fraction(factorial(argument)) ** factor.power
    total *= term.n_base.as_fraction() ** n * term.k_base.as_fraction() ** k
    return total


def test_binomial_quotients_match_exact_formula_and_direct_terms():
    term = _binomial_term()
    result = proper_hypergeometric_shift_quotients(term)
    for n, k in ((5, 2), (8, 3), (12, 7)):
        source = _direct(term, n, k)
        assert _eval_rf(result.n_ratio, n, k) == _direct(term, n + 1, k) / source
        assert _eval_rf(result.k_ratio, n, k) == _direct(term, n, k + 1) / source

    assert _eval_rf(result.n_ratio, 5, 2) == Fraction(3, 2)
    assert _eval_rf(result.k_ratio, 5, 2) == 1


def test_shifted_factorial_and_polynomial_prefactor_match_direct_evaluation():
    term = ProperHypergeometricTerm(
        polynomial=polynomial([((1, 0), 1), ((0, 1), 2), ((0, 0), 1)]),
        factorial_factors=(
            IntegerAffineFactorial(
                n_coefficient=1, k_coefficient=2, offset=3, power=-1
            ),
            IntegerAffineFactorial(n_coefficient=2, k_coefficient=1, offset=1, power=1),
        ),
        n_base=CanonicalRational(num=2, den=3),
        k_base=CanonicalRational(num=3, den=2),
    )
    result = proper_hypergeometric_shift_quotients(term)
    n, k = 2, 1
    source = _direct(term, n, k)
    assert _eval_rf(result.n_ratio, n, k) == _direct(term, n + 1, k) / source
    assert _eval_rf(result.k_ratio, n, k) == _direct(term, n, k + 1) / source


def test_zero_term_has_no_generic_shift_quotients():
    term = ProperHypergeometricTerm(polynomial=polynomial([]))
    with pytest.raises(OperationDomainValidationError, match="zero term"):
        proper_hypergeometric_shift_quotients(term)


def test_factorial_growth_is_rejected_before_expansion():
    term = ProperHypergeometricTerm(
        polynomial=polynomial([((0, 0), 1)]),
        factorial_factors=(
            IntegerAffineFactorial(
                n_coefficient=128, k_coefficient=0, offset=0, power=32
            ),
        ),
    )
    with pytest.raises(OperationResourceAdmissionError):
        proper_hypergeometric_shift_quotients(term)


def test_distinct_denominators_are_bounded_before_expansion():
    """Charge common-denominator growth across every prefactor coefficient.

    Sixteen distinct 10-digit prime denominators keep the largest individual
    width at 10 digits but grow the normalized quotient's common denominator to
    about 150 digits, so the request must be refused by admission rather than
    by the ``RationalFunction`` coefficient validator.
    """

    from sympy import nextprime

    primes: list[int] = []
    candidate = 10**9
    for _ in range(16):
        candidate = int(nextprime(candidate))
        primes.append(candidate)
        candidate += 1
    terms = tuple(
        RationalPolynomialTerm(
            coefficient=CanonicalRational(num=1, den=primes[degree]),
            exponents=(0, degree),
        )
        for degree in range(15, -1, -1)
    )
    term = ProperHypergeometricTerm(
        polynomial=RationalPolynomial(
            variables=("n", "k"),
            polynomial=SparseRationalPolynomial(terms=terms),
        )
    )
    with pytest.raises(OperationResourceAdmissionError):
        proper_hypergeometric_shift_quotients(term)


@pytest.mark.parametrize("component", ("prefactor", "n_base", "k_base"))
def test_oversized_exact_components_raise_admission_error(component):
    """Count digit widths without Python's integer-to-string conversion guard.

    ``CanonicalRational`` admits components well beyond the built-in 4,300-digit
    string limit, so admission must measure such widths without formatting the
    integer and reject the request through the shared resource envelope.
    """

    oversized = 10**5000 + 12345
    kwargs = {}
    terms = [((0, 0), 1)]
    if component == "prefactor":
        terms = [((0, 0), oversized)]
    elif component == "n_base":
        kwargs["n_base"] = CanonicalRational(num=oversized, den=1)
    else:
        kwargs["k_base"] = CanonicalRational(num=1, den=oversized)
    term = ProperHypergeometricTerm(
        polynomial=polynomial(terms),
        factorial_factors=(
            IntegerAffineFactorial(n_coefficient=1, k_coefficient=0, offset=0, power=1),
        ),
        **kwargs,
    )
    with pytest.raises(OperationResourceAdmissionError):
        proper_hypergeometric_shift_quotients(term)


def test_unbounded_affine_offset_is_charged_before_expansion():
    """Charge an uncapped affine offset against the output digit envelope.

    ``IntegerAffineFactorial.offset`` accepts arbitrary-size integers, so a
    many-thousand-digit offset would otherwise survive admission and fail inside
    the ``RationalFunction`` coefficient validator during cancellation.
    """

    term = ProperHypergeometricTerm(
        polynomial=polynomial([((0, 0), 1)]),
        factorial_factors=(
            IntegerAffineFactorial(
                n_coefficient=1,
                k_coefficient=0,
                offset=10**5000 + 12345,
                power=1,
            ),
        ),
    )
    with pytest.raises(OperationResourceAdmissionError):
        proper_hypergeometric_shift_quotients(term)
