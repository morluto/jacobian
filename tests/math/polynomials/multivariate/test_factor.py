"""Tests for multivariate polynomial factorization (#2105)."""

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.polynomials.multivariate._models import (
    MultivariateFactorRequest,
    MultivariateFactorResult,
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
        """(x*y -1)^2 = x^2*y^2 -2*x*y +1 should have multiplicity 2."""
        poly = _poly(("x", "y"), ((1, 1, (2, 2)), (-2, 1, (1, 1)), (1, 1, (0, 0))))
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


class TestMultivariateFactorResultInvariants:
    def test_roundtrip_result_validates(self):
        poly = _poly(("x", "y"), ((1, 1, (2, 1)), (-1, 1, (1, 0))))
        result = multivariate_factor(MultivariateFactorRequest(polynomial=poly))
        assert MultivariateFactorResult.model_validate(result.model_dump()) == result

    def test_rejects_zero_coefficient_with_zero_reconstruction(self):
        """Zero coefficient plus zero reconstruction must not validate."""
        zero = _poly(("x", "y"), ())
        with pytest.raises(ValidationError, match="coefficient must be nonzero"):
            MultivariateFactorResult(
                coefficient=CanonicalRational.from_fraction(Fraction(0)),
                factors=(),
                reconstructed=zero,
            )
        with pytest.raises(ValidationError, match="coefficient must be nonzero"):
            MultivariateFactorResult(
                coefficient=CanonicalRational.from_fraction(Fraction(0)),
                factors=(),
                reconstructed=_poly(("x", "y"), ((3, 2, (1, 1)),)),
            )

    def test_rejects_zero_reconstructed_polynomial(self):
        zero = _poly(("x", "y"), ())
        with pytest.raises(ValidationError, match="must be nonzero"):
            MultivariateFactorResult(
                coefficient=CanonicalRational.from_fraction(Fraction(1)),
                factors=(),
                reconstructed=zero,
            )

    def test_rejects_product_mismatch(self):
        reconstructed = _poly(("x", "y"), ((1, 1, (2, 0)),))
        with pytest.raises(ValidationError, match="does not equal reconstructed"):
            MultivariateFactorResult(
                coefficient=CanonicalRational.from_fraction(Fraction(2)),
                factors=(),
                reconstructed=reconstructed,
            )
        with pytest.raises(ValidationError, match="does not equal reconstructed"):
            MultivariateFactorResult(
                coefficient=CanonicalRational.from_fraction(Fraction(1)),
                factors=(),
                reconstructed=reconstructed,
            )


class TestOutputBudgetOutcome:
    def test_oversized_irreducible_factor_returns_typed_outcome(self):
        """(x^64-1)(y^64-1) + z(x-1)(y-1) factors with an irreducible factor
        of 4,097 terms; the request must return the typed budget outcome, not
        a host exception (review counterexample)."""
        # (x^64 - 1)*(y^64 - 1) + z*(x-1)*(y-1), descending lex order
        poly = _poly(
            ("x", "y", "z"),
            (
                (1, 1, (64, 64, 0)),
                (-1, 1, (64, 0, 0)),
                (1, 1, (1, 1, 1)),
                (-1, 1, (1, 0, 1)),
                (-1, 1, (0, 64, 0)),
                (-1, 1, (0, 1, 1)),
                (1, 1, (0, 0, 1)),
                (1, 1, (0, 0, 0)),
            ),
        )
        result = multivariate_factor(MultivariateFactorRequest(polynomial=poly))
        assert result.status == "OUTPUT_BUDGET_EXCEEDED"
        assert result.factors == ()
        assert result.reconstructed == poly
        assert MultivariateFactorResult.model_validate(
            result.model_dump()
        ) == result

    def test_budget_exceeded_claim_must_replay(self):
        """An authored OUTPUT_BUDGET_EXCEEDED label on a polynomial whose
        exact factorization fits the output budget must not validate."""
        poly = _poly(("x", "y"), ((1, 1, (2, 1)), (-1, 1, (1, 0))))
        with pytest.raises(ValidationError, match="not reproduced"):
            MultivariateFactorResult(
                status="OUTPUT_BUDGET_EXCEEDED",
                coefficient=CanonicalRational.from_fraction(Fraction(1)),
                factors=(),
                reconstructed=poly,
                normalization=None,
                product_reconstruction=None,
            )

    def test_budget_exceeded_cannot_carry_factors(self):
        poly = _poly(("x", "y"), ((2, 1, (2, 1)), (-2, 1, (1, 0))))
        with pytest.raises(ValidationError, match="carry no irreducible factors"):
            MultivariateFactorResult(
                status="OUTPUT_BUDGET_EXCEEDED",
                coefficient=CanonicalRational.from_fraction(Fraction(2)),
                factors=(
                    __import__(
                        "jacobian.math.polynomials.multivariate._models",
                        fromlist=["MultivariateIrreducibleFactor"],
                    ).MultivariateIrreducibleFactor(
                        factor=_poly(("x", "y"), ((1, 1, (1, 1)), (-1, 1, (1, 0)))),
                        multiplicity=1,
                    ),
                ),
                reconstructed=poly,
            )


class TestAggregateDegreeGate:
    def test_forged_aggregate_degree_rejected_before_expansion(self):
        """128 linear factors at multiplicity 64 cannot multiply against a
        small reconstruction; the degree gate rejects without expansion."""
        from jacobian.math.polynomials.multivariate._models import (
            MultivariateIrreducibleFactor,
        )

        factor = _poly(("x", "y"), ((1, 1, (1, 1)),))
        small = _poly(("x", "y"), ((1, 1, (0, 0)),))
        payload = tuple(
            MultivariateIrreducibleFactor(factor=factor, multiplicity=1)
            for _ in range(4)
        )
        with pytest.raises(
            ValidationError, match="aggregate irreducible degree exceeds"
        ):
            MultivariateFactorResult(
                coefficient=CanonicalRational.from_fraction(Fraction(1)),
                factors=payload,
                reconstructed=small,
            )
