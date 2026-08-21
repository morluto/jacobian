"""Tests for multivariate polynomial factorization (#2105)."""

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.polynomials.multivariate._models import (
    MultivariateFactorRequest,
)
from jacobian.math.polynomials.multivariate._operations import multivariate_factor
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _poly(variables, terms):
    return RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_fraction(Fraction(num, den)),
                    exponents=exps,
                )
                for num, den, exps in terms
            )
        ),
    )


class TestMultivariateFactor:
    def test_simple_factorization(self):
        """Factor x^2*y - x = x * (x*y - 1) in Q[x,y]."""
        poly = _poly(("x", "y"), ((1, 1, (2, 1)), (-1, 1, (1, 0))))
        result = multivariate_factor(MultivariateFactorRequest(polynomial=poly))
        assert len(result.factors) >= 1
        assert result.reconstructed is not None

    def test_irreducible(self):
        """An irreducible polynomial has one factor."""
        poly = _poly(("x", "y"), ((1, 1, (1, 1)), (-1, 1, (0, 0))))
        result = multivariate_factor(MultivariateFactorRequest(polynomial=poly))
        assert len(result.factors) == 1
        assert result.factors[0].multiplicity == 1

    def test_repeated_factor(self):
        """x^2 - 2*x + 1 = (x-1)^2 should have multiplicity 2."""
        poly = _poly(("x",), ((1, 1, (2,)), (-2, 1, (1,)), (1, 1, (0,))))
        result = multivariate_factor(MultivariateFactorRequest(polynomial=poly))
        assert len(result.factors) >= 1
        mults = [f.multiplicity for f in result.factors]
        assert 2 in mults

    def test_trivariate(self):
        """Factor x*y*z in Q[x,y,z]."""
        poly = _poly(("x", "y", "z"), ((1, 1, (1, 1, 1)),))
        result = multivariate_factor(MultivariateFactorRequest(polynomial=poly))
        assert len(result.factors) >= 1

    def test_constant_polynomial(self):
        """A constant has zero factors."""
        poly = _poly(("x", "y"), ((5, 1, (0, 0)),))
        result = multivariate_factor(MultivariateFactorRequest(polynomial=poly))
        assert len(result.factors) == 0
