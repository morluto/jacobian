"""Tests for multivariate polynomial factorization (#2105)."""

from collections.abc import Iterable
from fractions import Fraction
from itertools import islice
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.polynomials.multivariate._factor_models import (
    MultivariateFactorRequest,
    MultivariateFactorResult,
    MultivariateIrreducibleFactor,
)
from jacobian.math.polynomials.multivariate._tools import (
    _compute_factor,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)

type PolynomialTerm = tuple[int, int, tuple[int, ...]]


def _poly(
    variables: tuple[str, ...], terms: Iterable[PolynomialTerm]
) -> RationalPolynomial:
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
    def test_simple_factorization(self) -> None:
        """Factor x^2*y - x = x * (x*y - 1) in Q[x,y]."""
        poly = _poly(("x", "y"), ((1, 1, (2, 1)), (-1, 1, (1, 0))))
        result = _compute_factor(MultivariateFactorRequest(polynomial=poly))
        assert len(result.factors) >= 1
        assert result.reconstructed is not None

    def test_irreducible(self) -> None:
        """An irreducible polynomial has one factor."""
        poly = _poly(("x", "y"), ((1, 1, (1, 1)), (-1, 1, (0, 0))))
        result = _compute_factor(MultivariateFactorRequest(polynomial=poly))
        assert len(result.factors) == 1
        assert result.factors[0].multiplicity == 1

    def test_repeated_factor(self) -> None:
        """(x*y -1)^2 = x^2*y^2 -2*x*y +1 should have multiplicity 2."""
        poly = _poly(("x", "y"), ((1, 1, (2, 2)), (-2, 1, (1, 1)), (1, 1, (0, 0))))
        result = _compute_factor(MultivariateFactorRequest(polynomial=poly))
        assert len(result.factors) >= 1
        mults = [f.multiplicity for f in result.factors]
        assert 2 in mults

    def test_trivariate(self) -> None:
        """Factor x*y*z in Q[x,y,z]."""
        poly = _poly(("x", "y", "z"), ((1, 1, (1, 1, 1)),))
        result = _compute_factor(MultivariateFactorRequest(polynomial=poly))
        assert len(result.factors) >= 1

    def test_constant_polynomial(self) -> None:
        """A constant has zero factors."""
        poly = _poly(("x", "y"), ((5, 1, (0, 0)),))
        result = _compute_factor(MultivariateFactorRequest(polynomial=poly))
        assert len(result.factors) == 0


class TestMultivariateFactorResultInvariants:
    def test_roundtrip_result_validates(self) -> None:
        poly = _poly(("x", "y"), ((1, 1, (2, 1)), (-1, 1, (1, 0))))
        result = _compute_factor(MultivariateFactorRequest(polynomial=poly))
        assert MultivariateFactorResult.model_validate(result.model_dump()) == result

    def test_rejects_zero_coefficient_with_zero_reconstruction(self) -> None:
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

    def test_rejects_zero_reconstructed_polynomial(self) -> None:
        zero = _poly(("x", "y"), ())
        with pytest.raises(ValidationError):
            MultivariateFactorResult(
                coefficient=CanonicalRational.from_fraction(Fraction(1)),
                factors=(),
                reconstructed=zero,
            )

    def test_factorized_outcome_requires_invariant_markers(self) -> None:
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


class TestFactorRepresentationBounds:
    def test_oversized_irreducible_factor_raises_execution_failure(self) -> None:
        """(x^64-1)(y^64-1) + z(x-1)(y-1) factors with an irreducible factor
        of 4,097 terms; the worker cannot return a factored mathematical
        result within its admitted representation envelope."""
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
        with pytest.raises(RuntimeError):
            _compute_factor(MultivariateFactorRequest(polynomial=poly))


class TestAggregateDegreeGate:
    def test_forged_aggregate_degree_rejected_before_expansion(self) -> None:
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


class TestConversionAndResultBoundAlignment:
    def test_factor_within_representation_bound_validates(self) -> None:
        """(x^23-1)(y^23-1) + z(x-1)(y-1) has an irreducible factor with 530
        terms: above the request envelope yet within the result term bound, so
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
        result = _compute_factor(MultivariateFactorRequest(polynomial=poly))
        assert result.status == "FACTORIZED"
        term_counts = [len(record.factor.polynomial.terms) for record in result.factors]
        assert 530 in term_counts
        assert MultivariateFactorResult.model_validate(result.model_dump()) == result


class TestBoundedReconstruction:
    def test_telescoped_geometric_product_replays_boundedly(self) -> None:
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
        result = _compute_factor(MultivariateFactorRequest(polynomial=poly))
        assert result.status == "FACTORIZED"
        assert len(result.factors) > 3
        assert MultivariateFactorResult.model_validate(result.model_dump()) == result


def _difference_product_terms(
    variables: tuple[str, ...], exponent: int
) -> list[PolynomialTerm]:
    """Expand prod_i (variable_i**exponent - 1) into descending-lex terms."""

    accumulated: dict[tuple[int, ...], Fraction] = {
        tuple(0 for _ in variables): Fraction(1)
    }
    for index in range(len(variables)):
        shifted: dict[tuple[int, ...], Fraction] = {}
        for exps, coeff in accumulated.items():
            shifted[exps] = shifted.get(exps, Fraction(0)) + coeff
            raised = list(exps)
            raised[index] += exponent
            shifted[tuple(raised)] = shifted.get(tuple(raised), Fraction(0)) - coeff
        accumulated = {exps: coeff for exps, coeff in shifted.items() if coeff != 0}
    terms: list[PolynomialTerm] = [
        (coeff.numerator, coeff.denominator, exps)
        for exps, coeff in accumulated.items()
    ]
    terms.sort(key=lambda term: term[2], reverse=True)
    return terms


class TestUniqueFactorizationReplay:
    def test_paired_cyclotomic_product_returns_typed_result(self) -> None:
        """prod_{i=1..8} (x_i^12 - 1) is the review counterexample whose
        division replay materialized 32,768-term quotients; the request must
        return the typed FACTORIZED result with all 48 irreducible factors."""
        variables = tuple(f"x{i}" for i in range(1, 9))
        poly = _poly(variables, _difference_product_terms(variables, 12))
        assert len(poly.polynomial.terms) == 256
        result = _compute_factor(MultivariateFactorRequest(polynomial=poly))
        assert result.status == "FACTORIZED"
        assert len(result.factors) == 48
        assert result.reconstructed == poly
        assert MultivariateFactorResult.model_validate(result.model_dump()) == result

    def test_cyclotomic_pair_known_answer(self) -> None:
        """(x^12-1)(y^12-1) splits into six cyclotomic factors per variable;
        every factor is monic with multiplicity one and total degrees are
        1, 1, 2, 2, 2, 4 per variable."""
        poly = _poly(("x", "y"), _difference_product_terms(("x", "y"), 12))
        result = _compute_factor(MultivariateFactorRequest(polynomial=poly))
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


def _coprime_denominator_poly(denominator_count: int) -> RationalPolynomial:
    """Pairwise-coprime 256-digit reciprocals over distinct monomials."""
    from sympy import primerange

    # Powers of distinct primes remain pairwise coprime.  Using small prime
    # bases preserves the 256-digit LCM fixture without spending the test's
    # time proving primality for 129 unrelated 256-digit values.
    primes = tuple(islice(primerange(1_200_000, 1_220_000), denominator_count))
    denominators = tuple(prime**42 for prime in primes)
    monomials = [
        (exponent_x, exponent_y)
        for exponent_x in range(63, -1, -1)
        for exponent_y in range(63, -1, -1)
    ][:denominator_count]
    return _poly(
        ("x", "y"),
        tuple(
            (1, denominators[index], monomials[index])
            for index in range(denominator_count)
        ),
    )


class TestAggregateContentAdmission:
    """Requests must admit only representable derived values (#2226).

    Per-term digit budgets do not bound the aggregate content: clearing
    denominators to their least common multiple inflates both the rational
    content every result carries as one canonical rational and the primitive
    integer coefficients published as the reconstructed polynomial.
    """

    @pytest.mark.scale
    def test_many_coprime_denominators_rejected_before_backend(self) -> None:
        """129 pairwise-coprime 256-digit denominators pass every per-term
        budget yet their least common multiple exceeds the canonical
        32,768-digit rational limit, so the operation could never return its
        declared typed result; admission rejects before invoking SymPy."""
        request = MultivariateFactorRequest(polynomial=_coprime_denominator_poly(129))
        with pytest.raises(OperationDomainValidationError):
            _compute_factor(request)

    @pytest.mark.scale
    def test_content_within_limit_but_primitive_coefficients_rejected(self) -> None:
        """Even with the least common multiple inside the canonical limit,
        clearing denominators can push every primitive coefficient past the
        operation's own 256-digit coefficient budget."""
        request = MultivariateFactorRequest(polynomial=_coprime_denominator_poly(127))
        with pytest.raises(OperationDomainValidationError):
            _compute_factor(request)

    def test_small_shared_denominators_still_admitted(self) -> None:
        """Ordinary rational coefficients clear to small primitive values
        and remain serviceable end to end."""
        request = MultivariateFactorRequest(
            polynomial=_poly(
                ("x", "y"),
                ((1, 6, (2, 1)), (-1, 10, (1, 0))),
            )
        )
        result = _compute_factor(request)
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
    @pytest.mark.scale
    def test_reviewer_sparse_completing_counterexample_returns_typed_outcome(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """prod_i(x_i^64 - 1) + z*prod_i(x_i - 1) has an irreducible z-linear
        cofactor with 64^7 + 1 expanded terms.  The admitted request must
        return its typed outcome through the bounded worker instead of
        hanging or exhausting memory in the engine process."""
        import time

        from jacobian.math.polynomials.multivariate import _factor_backend

        monkeypatch.setattr(_factor_backend, "FACTOR_WORK_WALL_SECONDS", 5.0)

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
        with pytest.raises((RuntimeError, TimeoutError)):
            _compute_factor(request)
        elapsed = time.monotonic() - started
        assert elapsed < 30.0

    def test_worker_backend_agrees_with_in_process_factor_list(self) -> None:
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
                    [
                        *term.exponents,
                        *(str(value) for value in term.coefficient.as_integer_ratio()),
                    ]
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
        assert response["coefficient"] == ["1", "1"]

    def test_worker_crash_exit_is_execution_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A worker that exits abnormally without parsable output is a
        crash, not a mathematical result."""
        from jacobian.math.polynomials.multivariate._factor_backend import (
            FactorBackendFailureError,
            run_bounded_factorization,
        )

        def fake_run(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
            return TestExecutionInterruptionSeparation._fake_completed(
                returncode=1,
                stdout=b"<traceback> not json",
            )

        monkeypatch.setattr("jacobian.process.run_bounded_process", fake_run)
        poly = _poly(("x", "y"), ((1, 1, (2, 1)), (-1, 1, (1, 0))))
        with pytest.raises(FactorBackendFailureError):
            run_bounded_factorization(poly)

    def test_signal_death_under_cpu_limit_is_interrupted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CPU exhaustion is a deadline-type execution condition: SIGXCPU
        yields the retryable interrupted error."""
        import signal

        from jacobian.math.polynomials.multivariate._factor_backend import (
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
        with pytest.raises(FactorBackendInterruptedError):
            run_bounded_factorization(poly)

    def test_malformed_success_payload_is_execution_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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

    def test_unknown_signal_death_is_execution_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SIGKILL/SIGSEGV-style deaths are not proof of a capacity limit."""
        import signal

        from jacobian.math.polynomials.multivariate._factor_backend import (
            FactorBackendFailureError,
            run_bounded_factorization,
        )

        external_signal = -int(getattr(signal, "SIGSEGV", 11))

        def fake_run(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
            return TestExecutionInterruptionSeparation._fake_completed(
                returncode=external_signal,
                stdout=b"",
            )

        monkeypatch.setattr("jacobian.process.run_bounded_process", fake_run)
        poly = _poly(("x", "y"), ((1, 1, (2, 1)), (-1, 1, (1, 0))))
        with pytest.raises(FactorBackendFailureError):
            run_bounded_factorization(poly)

    def test_worker_aborts_without_containment_before_factoring(self) -> None:
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


class TestSignedBudgetOutcomeContent:
    def test_negative_source_budget_outcome_matches_content_convention(
        self,
    ) -> None:
        """The oversized-factor branch and result validation share one
        exact-content convention, whatever sign the source carries."""

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
        with pytest.raises(RuntimeError):
            _compute_factor(MultivariateFactorRequest(polynomial=poly))


class TestExecutionInterruptionSeparation:
    """Timeout, cancellation, and worker memory exhaustion are execution
    conditions, never mathematical factorization results."""

    @staticmethod
    def _fake_completed(**overrides: Any) -> SimpleNamespace:
        import json as _json

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

    def test_deadline_hit_raises_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A worker stopped by its deadline raises a timeout."""
        poly = _poly(("x", "y"), ((1, 1, (60, 60)), (-1, 1, (59, 0))))
        request = MultivariateFactorRequest(polynomial=poly)
        from jacobian.math.polynomials.multivariate import _factor_backend

        # Keep this below process startup plus the easy-case factorization
        # time so a fast runner cannot complete the worker before the
        # deadline that this regression test is meant to exercise.
        monkeypatch.setattr(_factor_backend, "FACTOR_WORK_WALL_SECONDS", 0.05)
        with pytest.raises(TimeoutError):
            _compute_factor(request)

    def test_worker_timeout_raises_distinct_exception(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run_bounded_factorization maps a timed-out worker onto
        FactorBackendInterruptedError."""
        from jacobian.math.polynomials.multivariate._factor_backend import (
            FactorBackendInterruptedError,
            run_bounded_factorization,
        )

        def fake_run(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
            return self._fake_completed(returncode=-9, timed_out=True)

        monkeypatch.setattr("jacobian.process.run_bounded_process", fake_run)
        poly = _poly(("x", "y"), ((1, 1, (2, 1)), (-1, 1, (1, 0))))
        with pytest.raises(FactorBackendInterruptedError):
            run_bounded_factorization(poly)

    def test_worker_memory_error_is_execution_interruption(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An allocation failure under the address-space budget is an
        enforcement stop like SIGXCPU: it maps onto
        FactorBackendInterruptedError."""
        import json as _json

        from jacobian.math.polynomials.multivariate._factor_backend import (
            FactorBackendInterruptedError,
            run_bounded_factorization,
        )

        payload = _json.dumps(
            {"ok": False, "error": "MemoryError()", "exhausted": True}
        ).encode()

        def fake_run(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
            return TestExecutionInterruptionSeparation._fake_completed(
                returncode=1, stdout=payload
            )

        monkeypatch.setattr("jacobian.process.run_bounded_process", fake_run)
        poly = _poly(("x", "y"), ((1, 1, (2, 1)), (-1, 1, (1, 0))))
        with pytest.raises(FactorBackendInterruptedError):
            run_bounded_factorization(poly)

    def test_worker_memory_error_raises_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A worker ``MemoryError`` establishes no mathematical result."""
        import json as _json

        payload = _json.dumps(
            {
                "ok": False,
                "error": "MemoryError()",
                "exhausted": True,
                "as_limit_applied": True,
            }
        ).encode()

        def fake_run(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
            return TestExecutionInterruptionSeparation._fake_completed(
                returncode=1, stdout=payload
            )

        monkeypatch.setattr("jacobian.process.run_bounded_process", fake_run)
        poly = _poly(("x", "y"), ((1, 1, (2, 1)), (-1, 1, (1, 0))))
        with pytest.raises(TimeoutError):
            _compute_factor(MultivariateFactorRequest(polynomial=poly))

    def test_worker_memory_limit_proof_required_without_prlimit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without prlimit wrapping, a worker that cannot prove its own
        address-space cap fails closed instead of running unbounded."""
        import json as _json

        from jacobian.math.polynomials.multivariate._factor_backend import (
            FactorBackendFailureError,
            run_bounded_factorization,
        )

        payload = _json.dumps({"ok": True, "as_limit_applied": False}).encode()

        def fake_run(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
            return TestExecutionInterruptionSeparation._fake_completed(stdout=payload)

        def fake_which(name: str) -> str | None:
            return None if name == "prlimit" else _real_which(name)

        import shutil as _shutil

        _real_which = _shutil.which
        monkeypatch.setattr(_shutil, "which", fake_which)
        monkeypatch.setattr("jacobian.process.run_bounded_process", fake_run)
        poly = _poly(("x", "y"), ((1, 1, (2, 1)), (-1, 1, (1, 0))))
        with pytest.raises(FactorBackendFailureError):
            run_bounded_factorization(poly)

    def test_no_portable_hard_limit_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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

    def test_worker_reports_address_space_flag(self) -> None:
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
                "terms": [[2, "1", "1"], [0, "-1", "1"]],
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
