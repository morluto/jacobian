"""Tests for multivariate polynomial factorization (#2105)."""

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.polynomials.multivariate._factor_models import (
    MultivariateFactorRequest,
    MultivariateFactorResult,
    MultivariateIrreducibleFactor,
)
from jacobian.math.polynomials.multivariate._operations import (
    multivariate_factor,
    verify_multivariate_factor_result,
)
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

    def test_kernel_result_uses_one_worker_call_until_explicitly_verified(
        self, monkeypatch
    ):
        """Construction and deserialization do not replay the producing worker."""

        from jacobian.math.polynomials.multivariate import _factor_backend

        calls = 0
        original = _factor_backend.run_bounded_factorization

        def counted(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(_factor_backend, "run_bounded_factorization", counted)
        poly = _poly(("x", "y"), ((1, 1, (2, 1)), (-1, 1, (1, 0))))
        result = multivariate_factor(MultivariateFactorRequest(polynomial=poly))
        assert calls == 1
        restored = MultivariateFactorResult.model_validate(result.model_dump())
        assert restored == result
        assert calls == 1
        assert verify_multivariate_factor_result(restored)
        assert calls == 2

    def test_rejects_zero_coefficient_with_zero_reconstruction(self):
        """Zero coefficient plus zero reconstruction must not validate."""
        zero = _poly(("x", "y"), ())
        with pytest.raises(ValidationError):
            MultivariateFactorResult(
                coefficient=CanonicalRational.from_fraction(Fraction(0)),
                factors=(),
                reconstructed=zero,
            )
        with pytest.raises(ValidationError):
            MultivariateFactorResult(
                coefficient=CanonicalRational.from_fraction(Fraction(0)),
                factors=(),
                reconstructed=_poly(("x", "y"), ((3, 2, (1, 1)),)),
            )

    def test_rejects_zero_reconstructed_polynomial(self):
        zero = _poly(("x", "y"), ())
        with pytest.raises(ValidationError):
            MultivariateFactorResult(
                coefficient=CanonicalRational.from_fraction(Fraction(1)),
                factors=(),
                reconstructed=zero,
            )

    def test_structural_result_defers_product_mismatch_to_explicit_verifier(self):
        reconstructed = _poly(("x", "y"), ((1, 1, (2, 0)),))
        wrong_content = MultivariateFactorResult(
            coefficient=CanonicalRational.from_fraction(Fraction(2)),
            factors=(),
            reconstructed=reconstructed,
        )
        assert not verify_multivariate_factor_result(wrong_content)
        forged = MultivariateFactorResult(
            coefficient=CanonicalRational.from_fraction(Fraction(1)),
            factors=(),
            reconstructed=reconstructed,
        )
        assert not verify_multivariate_factor_result(forged)

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
            with pytest.raises(ValidationError):
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

    def test_budget_exceeded_claim_requires_explicit_replay(self):
        """An authored OUTPUT_BUDGET_EXCEEDED label on a polynomial whose
        exact factorization fits the output budget must not validate."""
        poly = _poly(("x", "y"), ((1, 1, (2, 1)), (-1, 1, (1, 0))))
        forged = MultivariateFactorResult(
            status="OUTPUT_BUDGET_EXCEEDED",
            coefficient=CanonicalRational.from_fraction(Fraction(1)),
            factors=(),
            reconstructed=poly,
            normalization=None,
            product_reconstruction=None,
        )
        assert not verify_multivariate_factor_result(forged)

    def test_budget_exceeded_cannot_carry_factors(self):
        poly = _poly(("x", "y"), ((2, 1, (2, 1)), (-2, 1, (1, 0))))
        with pytest.raises(ValidationError):
            MultivariateFactorResult(
                status="OUTPUT_BUDGET_EXCEEDED",
                coefficient=CanonicalRational.from_fraction(Fraction(2)),
                factors=(
                    MultivariateIrreducibleFactor(
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
        with pytest.raises(ValidationError):
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
        forged = MultivariateFactorResult(
            coefficient=CanonicalRational.from_fraction(Fraction(1)),
            factors=tuple(records),
            reconstructed=target,
        )
        assert not verify_multivariate_factor_result(forged)

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
        forged = MultivariateFactorResult(
            coefficient=CanonicalRational.from_fraction(Fraction(1)),
            factors=(factor,),
            reconstructed=reconstructed,
        )
        assert not verify_multivariate_factor_result(forged)

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
        forged = MultivariateFactorResult(
            coefficient=CanonicalRational.from_fraction(Fraction(2)),
            factors=records,
            reconstructed=reconstructed,
        )
        assert not verify_multivariate_factor_result(forged)
        accepted = MultivariateFactorResult(
            coefficient=CanonicalRational.from_fraction(Fraction(3)),
            factors=records,
            reconstructed=reconstructed,
        )
        assert accepted.product_reconstruction == "EXACT"
        assert verify_multivariate_factor_result(accepted)


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
        forged = MultivariateFactorResult(
            coefficient=result.coefficient,
            factors=tuple(records),
            reconstructed=result.reconstructed,
        )
        assert not verify_multivariate_factor_result(forged)

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
        forged = MultivariateFactorResult(
            coefficient=result.coefficient,
            factors=tuple(records),
            reconstructed=result.reconstructed,
        )
        assert not verify_multivariate_factor_result(forged)


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
        with pytest.raises(ValidationError):
            MultivariateFactorRequest(polynomial=_prime_denominator_poly(129))

    def test_content_within_limit_but_primitive_coefficients_rejected(self):
        """Even with the least common multiple inside the canonical limit,
        clearing denominators can push every primitive coefficient past the
        operation's own 256-digit coefficient budget."""
        with pytest.raises(ValidationError):
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


def _expanded_product(
    variables: tuple[str, ...],
    binomials: list[tuple[int, int, int]],
) -> dict[tuple[int, ...], int]:
    """Expand one product of sparse binomials c*x_i^e - c exactly."""

    from collections import defaultdict

    size = len(variables)
    accumulator = defaultdict(int)
    accumulator[tuple(0 for _ in variables)] = 1
    for exponent, coefficient, index in binomials:
        shifted = tuple(
            exponent if position == index else 0 for position in range(size)
        )
        updated: dict[tuple[int, ...], int] = defaultdict(int)
        for monomial, value in accumulator.items():
            shifted_key = tuple(a + b for a, b in zip(monomial, shifted, strict=True))
            updated[shifted_key] += value * coefficient
            updated[monomial] -= value * coefficient
        accumulator = defaultdict(
            int,
            {k: v for k, v in updated.items() if v != 0},
        )
    return dict(accumulator)


class TestKillableFactorBackend:
    def test_reviewer_sparse_completing_counterexample_returns_typed_outcome(
        self, monkeypatch
    ):
        """prod_i(x_i^64 - 1) + z*prod_i(x_i - 1) has an irreducible z-linear
        cofactor with 64^7 + 1 expanded terms.  The admitted request must
        return its typed outcome through the bounded worker instead of
        hanging or exhausting memory in the engine process."""
        import time

        from jacobian.math.polynomials.multivariate import _factor_backend

        monkeypatch.setattr(_factor_backend, "FACTOR_WORK_WALL_SECONDS", 5.0)
        monkeypatch.setattr(_factor_backend, "FACTOR_VERIFY_WALL_SECONDS", 15.0)

        dense = _expanded_product(
            ("x1", "x2", "x3", "x4", "x5", "x6", "x7", "z"),
            [(64, 1, i) for i in range(7)],
        )
        z_linear = _expanded_product(
            ("x1", "x2", "x3", "x4", "x5", "x6", "x7", "z"),
            [(1, 1, i) for i in range(7)],
        )
        merged: dict[tuple[int, ...], int] = dict(dense)
        for monomial, value in z_linear.items():
            shifted = tuple(
                7 if position == 7 else degree
                for position, degree in enumerate(monomial)
            )
            merged[shifted] = merged.get(shifted, 0) + value
        poly = _poly(
            ("x1", "x2", "x3", "x4", "x5", "x6", "x7", "z"),
            tuple(
                (value, 1, exponents)
                for exponents, value in sorted(merged.items(), reverse=True)
                if value != 0
            ),
        )
        assert len(poly.polynomial.terms) <= 512

        request = MultivariateFactorRequest(polynomial=poly)
        started = time.monotonic()
        result = multivariate_factor(request)
        elapsed = time.monotonic() - started
        # Either status is exact: an output-capacity hit is the
        # mathematical bounded status, while a deadline, cancellation, or
        # memory-cap kill is the distinct retryable execution failure.
        # Both carry no factors and roundtrip.
        assert result.status in ("OUTPUT_BUDGET_EXCEEDED", "EXECUTION_FAILED")
        assert result.factors == ()
        assert result.normalization is None
        assert result.reconstructed == poly
        assert elapsed < 30.0
        assert MultivariateFactorResult.model_validate(result.model_dump()) == result

    def test_worker_backend_agrees_with_in_process_factor_list(self):
        """The bounded worker returns the same exact decomposition as an
        in-process ``factor_list`` on ordinary inputs."""
        import json
        import os
        import subprocess
        import sys

        from jacobian.math.polynomials.multivariate._factor_backend import (
            _WORKER_PATH,
        )

        poly = _poly(("x", "y"), ((1, 1, (2, 1)), (-1, 1, (1, 0))))
        payload = json.dumps(
            {
                "variables": ["x", "y"],
                "terms": [
                    [*term.exponents, *term.coefficient.as_integer_ratio()]
                    for term in poly.polynomial.terms
                ],
            }
        ).encode()
        completed = subprocess.run(
            [sys.executable, str(_WORKER_PATH)],
            input=payload,
            capture_output=True,
            timeout=60,
            env={
                **os.environ,
                "JACOBIAN_FACTOR_ADDRESS_SPACE_BYTES": str(4 * 1024 * 1024 * 1024),
            },
        )
        assert completed.returncode == 0
        response = json.loads(completed.stdout.decode())
        assert response["ok"] is True
        # x^2*y - x = x*(x*y - 1): two irreducible factors, content 1.
        assert len(response["factors"]) == 2
        assert response["coefficient"] == [1, 1]

    def test_worker_crash_exit_is_execution_failure(self, monkeypatch):
        """A worker that exits abnormally without parsable output is a
        crash, never an output-capacity conclusion."""
        from jacobian.math.polynomials.multivariate._factor_backend import (
            FactorBackendExhaustedError,
            FactorBackendFailureError,
            run_bounded_factorization,
        )

        def fake_run(*_args, **_kwargs):
            return TestExecutionInterruptionSeparation._fake_completed(
                returncode=1,
                stdout=b"<traceback> not json",
            )

        monkeypatch.setattr("jacobian.process.run_bounded_process", fake_run)
        poly = _poly(("x", "y"), ((1, 1, (2, 1)), (-1, 1, (1, 0))))
        with pytest.raises(FactorBackendFailureError) as exc_info:
            run_bounded_factorization(poly)
        assert not isinstance(exc_info.value, FactorBackendExhaustedError)

    def test_signal_death_under_cpu_limit_is_interrupted(self, monkeypatch):
        """CPU exhaustion is a deadline-type execution condition: SIGXCPU
        yields the retryable interrupted error, never a capacity status."""
        import signal

        from jacobian.math.polynomials.multivariate._factor_backend import (
            FactorBackendExhaustedError,
            FactorBackendInterruptedError,
            run_bounded_factorization,
        )

        sigxcpu = getattr(signal, "SIGXCPU", None)
        if sigxcpu is None:
            pytest.skip("POSIX-only signal semantics")
        monkeypatch.setattr(
            "jacobian.process.run_bounded_process",
            lambda *_a, **_k: TestExecutionInterruptionSeparation._fake_completed(
                returncode=-int(sigxcpu),
                stdout=b"",
            ),
        )
        poly = _poly(("x", "y"), ((1, 1, (2, 1)), (-1, 1, (1, 0))))
        with pytest.raises(FactorBackendInterruptedError) as exc_info:
            run_bounded_factorization(poly)
        assert not isinstance(exc_info.value, FactorBackendExhaustedError)

    def test_malformed_success_payload_is_execution_failure(self, monkeypatch):
        """A syntactically valid ok:true payload with a malformed result
        shape is a worker defect, not an exact decomposition."""
        import json as _json

        from jacobian.math.polynomials.multivariate._factor_backend import (
            FactorBackendFailureError,
            run_bounded_factorization,
        )

        payload = _json.dumps({"ok": True, "as_limit_applied": True}).encode()

        monkeypatch.setattr(
            "jacobian.process.run_bounded_process",
            lambda *_a, **_k: TestExecutionInterruptionSeparation._fake_completed(
                stdout=payload,
            ),
        )
        poly = _poly(("x", "y"), ((1, 1, (2, 1)), (-1, 1, (1, 0))))
        with pytest.raises(FactorBackendFailureError, match="malformed"):
            run_bounded_factorization(poly)

    def test_unknown_signal_death_is_execution_failure(self, monkeypatch):
        """SIGKILL/SIGSEGV-style deaths are not proof of a capacity limit."""
        import signal

        from jacobian.math.polynomials.multivariate._factor_backend import (
            FactorBackendExhaustedError,
            FactorBackendFailureError,
            run_bounded_factorization,
        )

        external_signal = -int(getattr(signal, "SIGSEGV", 11))

        def fake_run(*_args, **_kwargs):
            return TestExecutionInterruptionSeparation._fake_completed(
                returncode=external_signal,
                stdout=b"",
            )

        monkeypatch.setattr("jacobian.process.run_bounded_process", fake_run)
        poly = _poly(("x", "y"), ((1, 1, (2, 1)), (-1, 1, (1, 0))))
        with pytest.raises(FactorBackendFailureError) as exc_info:
            run_bounded_factorization(poly)
        assert not isinstance(exc_info.value, FactorBackendExhaustedError)

    def test_worker_aborts_without_containment_before_factoring(self):
        """A worker that cannot apply its address-space cap exits before
        any allocation-heavy factorization work."""
        import json
        import os
        import subprocess
        import sys

        from jacobian.math.polynomials.multivariate._factor_backend import (
            _WORKER_PATH,
        )

        payload = json.dumps({"variables": ["x"], "terms": [[2, 1, 1]]}).encode()
        environment = dict(os.environ)
        # No JACOBIAN_FACTOR_ADDRESS_SPACE_BYTES: the worker must refuse.
        environment.pop("JACOBIAN_FACTOR_ADDRESS_SPACE_BYTES", None)
        completed = subprocess.run(
            [sys.executable, str(_WORKER_PATH)],
            input=payload,
            capture_output=True,
            timeout=60,
            env=environment,
        )
        assert completed.returncode == 1
        response = json.loads(completed.stdout.decode())
        assert response["ok"] is False
        assert response["exhausted"] is False
        assert response["as_limit_applied"] is False

    def test_interrupted_budget_claim_replay_rejected(self, monkeypatch):
        """An authored OUTPUT_BUDGET_EXCEEDED claim whose verification
        replay is interrupted must be rejected, not authenticated."""
        from jacobian.math.polynomials.multivariate._factor_backend import (
            FactorBackendInterruptedError,
        )

        poly = _poly(("x", "y"), ((2, 1, (2, 1)), (-2, 1, (1, 0))))

        def fake_run(*_args, **_kwargs):
            raise FactorBackendInterruptedError("replay stopped")

        monkeypatch.setattr(
            "jacobian.process.run_bounded_process",
            fake_run,
        )
        from jacobian._exact import CanonicalRational

        claim = MultivariateFactorResult(
            status="OUTPUT_BUDGET_EXCEEDED",
            coefficient=CanonicalRational.from_fraction(Fraction(2)),
            factors=(),
            reconstructed=poly,
            normalization=None,
            product_reconstruction=None,
        )
        assert not verify_multivariate_factor_result(claim)

    def test_memory_exhausted_budget_claim_replay_rejected(self, monkeypatch):
        """An authored OUTPUT_BUDGET_EXCEEDED claim whose verification
        replay itself dies on worker memory exhaustion must be rejected,
        not authenticated: an allocation failure under the address-space
        cap proves nothing about the exact output size (PR #2226 review)."""
        import json as _json

        poly = _poly(("x", "y"), ((2, 1, (2, 1)), (-2, 1, (1, 0))))

        payload = _json.dumps(
            {
                "ok": False,
                "error": "MemoryError()",
                "exhausted": True,
                "as_limit_applied": True,
            }
        ).encode()

        def fake_run(*_args, **_kwargs):
            return TestExecutionInterruptionSeparation._fake_completed(
                returncode=1, stdout=payload
            )

        monkeypatch.setattr("jacobian.process.run_bounded_process", fake_run)
        from jacobian._exact import CanonicalRational

        claim = MultivariateFactorResult(
            status="OUTPUT_BUDGET_EXCEEDED",
            coefficient=CanonicalRational.from_fraction(Fraction(2)),
            factors=(),
            reconstructed=poly,
            normalization=None,
            product_reconstruction=None,
        )
        assert not verify_multivariate_factor_result(claim)


class TestSignedBudgetOutcomeContent:
    def test_negative_source_budget_outcome_matches_content_convention(self):
        """The oversized-factor branch and result validation share one
        exact-content convention, whatever sign the source carries."""

        from jacobian.math.polynomials.multivariate import _factor_backend

        dense = _expanded_product(
            ("x", "y", "z"),
            [(64, 1, 0), (64, 1, 1)],
        )
        z_linear = _expanded_product(("x", "y", "z"), [(1, 1, 0), (1, 1, 1)])
        merged: dict[tuple[int, ...], int] = dict(dense)
        for monomial, value in z_linear.items():
            shifted = tuple(
                1 if position == 2 else degree
                for position, degree in enumerate(monomial)
            )
            merged[shifted] = merged.get(shifted, 0) + value
        merged = {monomial: -value for monomial, value in merged.items()}
        poly = _poly(
            ("x", "y", "z"),
            tuple(
                (value, 1, exponents)
                for exponents, value in sorted(merged.items(), reverse=True)
                if value != 0
            ),
        )
        result = multivariate_factor(MultivariateFactorRequest(polynomial=poly))
        if result.status != "OUTPUT_BUDGET_EXCEEDED":
            pytest.fail("expected the oversized-factor bounded outcome")
        expected = _factor_backend.primitive_content_fraction(poly)
        assert result.coefficient.as_fraction() == expected
        assert MultivariateFactorResult.model_validate(result.model_dump()) == result


class TestExecutionInterruptionSeparation:
    """Timeout, cancellation, and worker memory exhaustion are execution
    conditions, never the mathematical OUTPUT_BUDGET_EXCEEDED status
    (PR #2226 review)."""

    @staticmethod
    def _fake_completed(**overrides):
        import json as _json
        from types import SimpleNamespace

        defaults = {
            "returncode": 0,
            "stdout": _json.dumps(
                {
                    "ok": True,
                    "coefficient": [1, 1],
                    "factors": [],
                    "as_limit_applied": True,
                }
            ).encode(),
            "stderr": b"",
            "stdout_exceeded": False,
            "stderr_exceeded": False,
            "timed_out": False,
            "cancelled": False,
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_deadline_hit_returns_interrupted_not_budget_exceeded(self, monkeypatch):
        """A worker stopped by its deadline yields EXECUTION_FAILED,
        which validates without rerunning any factorization."""
        poly = _poly(("x", "y"), ((1, 1, (60, 60)), (-1, 1, (59, 0))))
        request = MultivariateFactorRequest(polynomial=poly)
        from jacobian.math.polynomials.multivariate import _factor_backend

        monkeypatch.setattr(_factor_backend, "FACTOR_WORK_WALL_SECONDS", 0.25)
        result = multivariate_factor(request)
        assert result.status == "EXECUTION_FAILED"
        assert result.factors == ()
        assert result.reconstructed == poly
        assert result.coefficient.as_fraction() == Fraction(1)
        # Roundtrip: an interruption claim establishes nothing to
        # reproduce, so validation must not rerun the kernel.
        assert MultivariateFactorResult.model_validate(result.model_dump()) == result

    def test_worker_timeout_raises_distinct_exception(self, monkeypatch):
        """run_bounded_factorization maps a timed-out worker onto
        FactorBackendInterruptedError rather than ExhaustedError."""
        from jacobian.math.polynomials.multivariate._factor_backend import (
            FactorBackendExhaustedError,
            FactorBackendInterruptedError,
            run_bounded_factorization,
        )

        def fake_run(*_args, **_kwargs):
            return self._fake_completed(returncode=-9, timed_out=True)

        monkeypatch.setattr("jacobian.process.run_bounded_process", fake_run)
        poly = _poly(("x", "y"), ((1, 1, (2, 1)), (-1, 1, (1, 0))))
        with pytest.raises(FactorBackendInterruptedError) as exc_info:
            run_bounded_factorization(poly)
        assert not isinstance(exc_info.value, FactorBackendExhaustedError)

    def test_worker_memory_error_is_execution_interruption(self, monkeypatch):
        """An allocation failure under the address-space budget is an
        enforcement stop like SIGXCPU: it maps onto
        FactorBackendInterruptedError, never a capacity status."""
        import json as _json

        from jacobian.math.polynomials.multivariate._factor_backend import (
            FactorBackendExhaustedError,
            FactorBackendInterruptedError,
            run_bounded_factorization,
        )

        payload = _json.dumps(
            {"ok": False, "error": "MemoryError()", "exhausted": True}
        ).encode()

        def fake_run(*_args, **_kwargs):
            return TestExecutionInterruptionSeparation._fake_completed(
                returncode=1, stdout=payload
            )

        monkeypatch.setattr("jacobian.process.run_bounded_process", fake_run)
        poly = _poly(("x", "y"), ((1, 1, (2, 1)), (-1, 1, (1, 0))))
        with pytest.raises(FactorBackendInterruptedError) as exc_info:
            run_bounded_factorization(poly)
        assert not isinstance(exc_info.value, FactorBackendExhaustedError)

    def test_worker_memory_error_returns_execution_failed_not_budget_exceeded(
        self, monkeypatch
    ):
        """A worker ``MemoryError`` under the address-space cap proves only
        that this run's work envelope was too small; it establishes nothing
        about the exact output size, so multivariate_factor must return the
        retryable EXECUTION_FAILED status, never OUTPUT_BUDGET_EXCEEDED
        (PR #2226 review)."""
        import json as _json

        from jacobian.math.polynomials.multivariate import _factor_backend

        payload = _json.dumps(
            {
                "ok": False,
                "error": "MemoryError()",
                "exhausted": True,
                "as_limit_applied": True,
            }
        ).encode()

        def fake_run(*_args, **_kwargs):
            return TestExecutionInterruptionSeparation._fake_completed(
                returncode=1, stdout=payload
            )

        monkeypatch.setattr("jacobian.process.run_bounded_process", fake_run)
        poly = _poly(("x", "y"), ((1, 1, (2, 1)), (-1, 1, (1, 0))))
        result = multivariate_factor(MultivariateFactorRequest(polynomial=poly))
        assert result.status == "EXECUTION_FAILED"
        assert result.factors == ()
        assert result.reconstructed == poly
        expected = _factor_backend.primitive_content_fraction(poly)
        assert result.coefficient.as_fraction() == expected
        # Roundtrip: an interruption claim establishes nothing to
        # reproduce, so validation must not rerun the kernel.
        assert MultivariateFactorResult.model_validate(result.model_dump()) == result

    def test_worker_memory_limit_proof_required_without_prlimit(self, monkeypatch):
        """Without prlimit wrapping, a worker that cannot prove its own
        address-space cap fails closed instead of running unbounded."""
        import json as _json

        from jacobian.math.polynomials.multivariate._factor_backend import (
            FactorBackendFailureError,
            run_bounded_factorization,
        )

        payload = _json.dumps({"ok": True, "as_limit_applied": False}).encode()

        def fake_run(*_args, **_kwargs):
            return TestExecutionInterruptionSeparation._fake_completed(stdout=payload)

        def fake_which(name):
            return None if name == "prlimit" else _real_which(name)

        import shutil as _shutil

        _real_which = _shutil.which
        monkeypatch.setattr(_shutil, "which", fake_which)
        monkeypatch.setattr("jacobian.process.run_bounded_process", fake_run)
        poly = _poly(("x", "y"), ((1, 1, (2, 1)), (-1, 1, (1, 0))))
        with pytest.raises(FactorBackendFailureError):
            run_bounded_factorization(poly)

    def test_no_portable_hard_limit_fails_closed(self, monkeypatch):
        """A platform with neither prlimit nor POSIX self-limiting is
        rejected before launching an unbounded worker."""
        import os as _os
        import shutil as _shutil

        from jacobian.math.polynomials.multivariate._factor_backend import (
            FactorBackendFailureError,
            run_bounded_factorization,
        )

        monkeypatch.setattr(_shutil, "which", lambda name: None)
        monkeypatch.setattr(_os, "name", "nt")
        poly = _poly(("x", "y"), ((1, 1, (2, 1)), (-1, 1, (1, 0))))
        with pytest.raises(FactorBackendFailureError):
            run_bounded_factorization(poly)

    def test_forged_interruption_coefficient_rejected(self):
        """An authored interruption claim must still bind its coefficient
        to the exact content of the restated polynomial."""
        from jacobian._exact import CanonicalRational

        poly = _poly(("x", "y"), ((1, 1, (2, 1)), (-1, 1, (1, 0))))
        with pytest.raises(ValidationError):
            MultivariateFactorResult(
                status="EXECUTION_FAILED",
                coefficient=CanonicalRational.from_fraction(Fraction(5)),
                factors=(),
                reconstructed=poly,
                normalization=None,
                product_reconstruction=None,
            )

    def test_worker_reports_address_space_flag(self):
        """The worker response carries its hard-limit proof on success."""
        import json
        import os
        import subprocess
        import sys

        from jacobian.math.polynomials.multivariate._factor_backend import (
            _WORKER_PATH,
        )

        payload = json.dumps(
            {
                "variables": ["x"],
                "terms": [[2, 1, 1], [0, -1, 1]],
            }
        ).encode()
        environment = dict(os.environ)
        environment["JACOBIAN_FACTOR_ADDRESS_SPACE_BYTES"] = str(4 * 1024 * 1024 * 1024)
        completed = subprocess.run(
            [sys.executable, str(_WORKER_PATH)],
            input=payload,
            capture_output=True,
            timeout=60,
            env=environment,
        )
        assert completed.returncode == 0
        response = json.loads(completed.stdout.decode())
        assert response["ok"] is True
        assert response["as_limit_applied"] is True
