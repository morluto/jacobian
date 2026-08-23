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

    def test_factorized_outcome_requires_invariant_markers(self):
        """A FACTORIZED result without the public contract's normalization
        and product-reconstruction literals cannot validate: consumers rely
        on those markers to interpret the decomposition."""
        reconstructed = _poly(("x", "y"), ((1, 1, (2, 0)),))
        for kwargs in (
            {"normalization": None, "product_reconstruction": "EXACT"},
            {
                "normalization": "CONTENT_AND_MONIC_IRREDUCIBLES",
                "product_reconstruction": None,
            },
            {"normalization": None, "product_reconstruction": None},
        ):
            with pytest.raises(ValidationError, match="FACTORIZED outcomes declare"):
                MultivariateFactorResult(
                    coefficient=CanonicalRational.from_fraction(Fraction(1)),
                    factors=(),
                    reconstructed=reconstructed,
                    **kwargs,
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


def _difference_product_terms(variables, exponent):
    """Expand prod_i (variable_i**exponent - 1) into descending-lex terms."""

    accumulated = {tuple(0 for _ in variables): Fraction(1)}
    for index in range(len(variables)):
        shifted = {}
        for exps, coeff in accumulated.items():
            shifted[exps] = shifted.get(exps, Fraction(0)) + coeff
            raised = list(exps)
            raised[index] += exponent
            shifted[tuple(raised)] = shifted.get(tuple(raised), Fraction(0)) - coeff
        accumulated = {exps: coeff for exps, coeff in shifted.items() if coeff != 0}
    terms = [
        (coeff.numerator, coeff.denominator, exps)
        for exps, coeff in accumulated.items()
    ]
    terms.sort(key=lambda term: term[2], reverse=True)
    return terms


class TestUniqueFactorizationReplay:
    def test_paired_cyclotomic_product_returns_typed_result(self):
        """prod_{i=1..8} (x_i^12 - 1) is the review counterexample whose
        division replay materialized 32,768-term quotients; the request must
        return the typed FACTORIZED result with all 48 irreducible factors."""
        variables = tuple(f"x{i}" for i in range(1, 9))
        poly = _poly(variables, _difference_product_terms(variables, 12))
        assert len(poly.polynomial.terms) == 256
        result = multivariate_factor(MultivariateFactorRequest(polynomial=poly))
        assert result.status == "FACTORIZED"
        assert len(result.factors) == 48
        assert result.reconstructed == poly
        assert MultivariateFactorResult.model_validate(result.model_dump()) == result

    def test_cyclotomic_pair_known_answer(self):
        """(x^12-1)(y^12-1) splits into six cyclotomic factors per variable;
        every factor is monic with multiplicity one and total degrees are
        1, 1, 2, 2, 2, 4 per variable."""
        poly = _poly(("x", "y"), _difference_product_terms(("x", "y"), 12))
        result = multivariate_factor(MultivariateFactorRequest(polynomial=poly))
        assert result.status == "FACTORIZED"
        assert result.coefficient.as_fraction() == 1
        assert {record.multiplicity for record in result.factors} == {1}
        degrees = sorted(
            max(
                (sum(term.exponents) for term in record.factor.polynomial.terms),
                default=0,
            )
            for record in result.factors
        )
        assert degrees == [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 4, 4]

    def test_swapped_monic_irreducible_factor_rejected(self):
        """Replacing one true factor with another monic irreducible of the
        same total degree keeps the canonical envelope but must fail the
        unique-factorization replay."""
        poly = _poly(("x", "y"), _difference_product_terms(("x", "y"), 12))
        result = multivariate_factor(MultivariateFactorRequest(polynomial=poly))
        impostor = MultivariateIrreducibleFactor(
            factor=_poly(("x", "y"), ((1, 1, (2, 0)), (1, 1, (0, 2)), (1, 1, (0, 0)))),
            multiplicity=1,
        )
        records = [*result.factors[:-1], impostor]
        records.sort(key=_sort_key)
        with pytest.raises(ValidationError, match="does not equal reconstructed"):
            MultivariateFactorResult(
                coefficient=result.coefficient,
                factors=tuple(records),
                reconstructed=result.reconstructed,
            )

    def test_multiplicity_shift_between_equal_degree_factors_rejected(self):
        """Moving multiplicity between equal-degree factors preserves the
        aggregate degree yet changes the decomposition; the replay must
        reject it."""
        poly = _poly(("x", "y"), _difference_product_terms(("x", "y"), 12))
        result = multivariate_factor(MultivariateFactorRequest(polynomial=poly))

        def degree_of(record):
            return max(
                (sum(term.exponents) for term in record.factor.polynomial.terms),
                default=0,
            )

        by_degree = {}
        for record in result.factors:
            by_degree.setdefault(degree_of(record), []).append(record)
        victims = by_degree[2]
        dropped, kept = victims[0], victims[1]
        promoted = MultivariateIrreducibleFactor(
            factor=victims[2].factor,
            multiplicity=2,
        )
        records = [
            record
            for record in result.factors
            if record not in (dropped, kept, victims[2])
        ]
        records.append(promoted)
        records.sort(key=_sort_key)
        with pytest.raises(ValidationError, match="does not equal reconstructed"):
            MultivariateFactorResult(
                coefficient=result.coefficient,
                factors=tuple(records),
                reconstructed=result.reconstructed,
            )


def _prime_denominator_poly(prime_count):
    """Distinct 256-digit prime reciprocals over distinct bivariate monomials."""
    from sympy import nextprime

    primes = []
    candidate = 10**255 + 12345
    for _ in range(prime_count):
        candidate = nextprime(candidate)
        primes.append(candidate)
    monomials = [
        (exponent_x, exponent_y)
        for exponent_x in range(63, -1, -1)
        for exponent_y in range(63, -1, -1)
    ][:prime_count]
    return _poly(
        ("x", "y"),
        tuple((1, primes[index], monomials[index]) for index in range(prime_count)),
    )


class TestAggregateContentAdmission:
    """Requests must admit only representable derived values (#2226).

    Per-term digit budgets do not bound the aggregate content: clearing
    denominators to their least common multiple inflates both the rational
    content every result carries as one canonical rational and the primitive
    integer coefficients published as the reconstructed polynomial.
    """

    def test_many_prime_denominators_rejected_before_backend(self):
        """129 distinct 256-digit prime denominators pass every per-term
        budget yet their least common multiple exceeds the canonical
        32,768-digit rational limit, so the operation could never return its
        declared typed result; admission rejects before invoking SymPy."""
        with pytest.raises(ValidationError, match="aggregate rational content"):
            MultivariateFactorRequest(polynomial=_prime_denominator_poly(129))

    def test_content_within_limit_but_primitive_coefficients_rejected(self):
        """Even with the least common multiple inside the canonical limit,
        clearing denominators can push every primitive coefficient past the
        operation's own 256-digit coefficient budget."""
        with pytest.raises(ValidationError, match="primitive integer coefficients"):
            MultivariateFactorRequest(polynomial=_prime_denominator_poly(127))

    def test_small_shared_denominators_still_admitted(self):
        """Ordinary rational coefficients clear to small primitive values
        and remain serviceable end to end."""
        request = MultivariateFactorRequest(
            polynomial=_poly(
                ("x", "y"),
                ((1, 6, (2, 1)), (-1, 10, (1, 0))),
            )
        )
        result = multivariate_factor(request)
        assert result.status == "FACTORIZED"
        assert MultivariateFactorResult.model_validate(result.model_dump()) == result
