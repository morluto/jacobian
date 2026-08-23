"""Tests for multivariate polynomial factorization (#2105)."""

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.polynomials.multivariate._models import (
    MultivariateFactorRequest,
    MultivariateFactorResult,
    MultivariateIrreducibleFactor,
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
        assert MultivariateFactorResult.model_validate(result.model_dump()) == result

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


def _sort_key(record):
    return (
        record.multiplicity,
        max(
            (sum(term.exponents) for term in record.factor.polynomial.terms), default=0
        ),
        tuple(
            (term.exponents, term.coefficient.num, term.coefficient.den)
            for term in record.factor.polynomial.terms
        ),
    )


class TestConversionAndResultLimitAlignment:
    def test_factor_within_output_budget_validates(self):
        """(x^23-1)(y^23-1) + z(x-1)(y-1) has an irreducible factor with 530
        terms: above the request envelope yet within the output budget, so
        the result must validate instead of leaking a host exception."""
        poly = _poly(
            ("x", "y", "z"),
            (
                (1, 1, (23, 23, 0)),
                (-1, 1, (23, 0, 0)),
                (1, 1, (1, 1, 1)),
                (-1, 1, (1, 0, 1)),
                (-1, 1, (0, 23, 0)),
                (-1, 1, (0, 1, 1)),
                (1, 1, (0, 0, 1)),
                (1, 1, (0, 0, 0)),
            ),
        )
        result = multivariate_factor(MultivariateFactorRequest(polynomial=poly))
        assert result.status == "FACTORIZED"
        term_counts = [len(record.factor.polynomial.terms) for record in result.factors]
        assert 530 in term_counts
        assert MultivariateFactorResult.model_validate(result.model_dump()) == result


class TestBoundedReconstructionReplay:
    def test_equal_degree_forged_payload_rejected_without_expansion(self):
        """64 distinct monic linear forms against v0^64 share the aggregate
        degree, so only the no-expansion replay can reject them; it must
        fail on the first inexact division instead of expanding."""
        variables = tuple(f"v{i}" for i in range(8))

        def exponents(assignment):
            return tuple(assignment.get(variable, 0) for variable in variables)

        records = []
        for index in range(64):
            low, high = sorted((index % 8, (index + 1) % 8))
            linear = _poly(
                variables,
                (
                    (1, 1, exponents({variables[low]: 1})),
                    (2, 1, exponents({variables[high]: 1})),
                    (index + 3, 1, exponents({})),
                ),
            )
            records.append(MultivariateIrreducibleFactor(factor=linear, multiplicity=1))
        records.sort(key=_sort_key)
        target = _poly(variables, ((1, 1, exponents({"v0": 64})),))
        with pytest.raises(ValidationError, match="does not equal reconstructed"):
            MultivariateFactorResult(
                coefficient=CanonicalRational.from_fraction(Fraction(1)),
                factors=tuple(records),
                reconstructed=target,
            )

    def test_telescoped_geometric_product_replays_boundedly(self):
        """(x^64-1)(y^64-1)(z^64-1) reconstructs through many geometric-sum
        factors; the division replay must verify it exactly and quickly."""
        poly = _poly(
            ("x", "y", "z"),
            (
                (1, 1, (64, 64, 64)),
                (-1, 1, (64, 64, 0)),
                (-1, 1, (64, 0, 64)),
                (1, 1, (64, 0, 0)),
                (-1, 1, (0, 64, 64)),
                (1, 1, (0, 64, 0)),
                (1, 1, (0, 0, 64)),
                (-1, 1, (0, 0, 0)),
            ),
        )
        result = multivariate_factor(MultivariateFactorRequest(polynomial=poly))
        assert result.status == "FACTORIZED"
        assert len(result.factors) > 3
        assert MultivariateFactorResult.model_validate(result.model_dump()) == result

    def test_nonzero_remainder_rejected_as_mismatch(self):
        """x^2 + y^2 shares its aggregate degree with (x+y) but division
        leaves a remainder, so the replay must reject it as a mismatch."""
        reconstructed = _poly(("x", "y"), ((1, 1, (2, 0)), (1, 1, (0, 2))))
        factor = MultivariateIrreducibleFactor(
            factor=_poly(("x", "y"), ((1, 1, (1, 1)), (1, 1, (0, 0)))),
            multiplicity=1,
        )
        with pytest.raises(ValidationError, match="does not equal reconstructed"):
            MultivariateFactorResult(
                coefficient=CanonicalRational.from_fraction(Fraction(1)),
                factors=(factor,),
                reconstructed=reconstructed,
            )

    def test_scaled_constant_coefficient_verified(self):
        """coefficient * product must equal reconstructed exactly, including
        rational content placement: 2*(x)*(y) != reconstructed 3*x*y."""
        records = (
            MultivariateIrreducibleFactor(
                factor=_poly(("x", "y"), ((1, 1, (0, 1)),)), multiplicity=1
            ),
            MultivariateIrreducibleFactor(
                factor=_poly(("x", "y"), ((1, 1, (1, 0)),)), multiplicity=1
            ),
        )
        reconstructed = _poly(("x", "y"), ((3, 1, (1, 1)),))
        with pytest.raises(ValidationError, match="does not equal reconstructed"):
            MultivariateFactorResult(
                coefficient=CanonicalRational.from_fraction(Fraction(2)),
                factors=records,
                reconstructed=reconstructed,
            )
        accepted = MultivariateFactorResult(
            coefficient=CanonicalRational.from_fraction(Fraction(3)),
            factors=records,
            reconstructed=reconstructed,
        )
        assert accepted.product_reconstruction == "EXACT"


class TestBudgetOutcomeCoefficientBinding:
    def test_budget_exceeded_outcome_rejects_altered_coefficient(self):
        """The typed OUTPUT_BUDGET_EXCEEDED outcome carries the exact rational
        content; revalidation must reject any other nonzero coefficient."""
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
        outcome = multivariate_factor(MultivariateFactorRequest(polynomial=poly))
        assert outcome.status == "OUTPUT_BUDGET_EXCEEDED"
        dump = outcome.model_dump()
        assert MultivariateFactorResult.model_validate(dump) == outcome
        for numerator in ("999", "-1", "0"):
            dump["coefficient"] = {"num": numerator, "den": "7"}
            with pytest.raises(ValidationError):
                MultivariateFactorResult.model_validate(dump)
