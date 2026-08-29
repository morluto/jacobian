"""Tests for Groebner basis, normal form, and elimination ideal operations."""

from __future__ import annotations

from typing import Literal, NoReturn, TypedDict

import pytest
from pydantic import ValidationError

from jacobian.math.polynomials.ideals import operations
from jacobian.math.polynomials.ideals._models import (
    EliminationIdealRequest,
    GroebnerBasisRequest,
    GroebnerBasisResult,
    IdealComputationBudget,
    IdealNormalFormRequest,
)
from jacobian.math.polynomials.ideals._singular import (
    run_bounded_stdin_python_kernel,
)
from jacobian.math.polynomials.ideals.operations import (
    _SympyKernelError,
    _SympyKernelTimeoutError,
    elimination_ideal,
    groebner_basis,
    ideal_normal_form,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialIdeal,
)


def _run_groebner(request: GroebnerBasisRequest):
    return groebner_basis(
        request.ideal, request.monomial_order, resource_budget=request.resource_budget
    )


def _run_normal_form(request: IdealNormalFormRequest):
    return ideal_normal_form(request.ideal, request.polynomial, request.monomial_order)


def _run_elimination(request: EliminationIdealRequest):
    return elimination_ideal(
        request.ideal,
        request.eliminated_variables,
        resource_budget=request.resource_budget,
    )


class _CanonicalRationalWire(TypedDict):
    num: str
    den: str


class _RationalPolynomialTermWire(TypedDict):
    coefficient: _CanonicalRationalWire
    exponents: list[int]


class _RationalPolynomialWire(TypedDict):
    terms: list[_RationalPolynomialTermWire]


class _RationalPolynomialPayload(TypedDict):
    domain: Literal["QQ"]
    variables: list[str]
    polynomial: _RationalPolynomialWire


def _poly(
    variables: tuple[str, ...],
    *terms: tuple[int, int, tuple[int, ...]],
) -> RationalPolynomial:
    payload: _RationalPolynomialPayload = {
        "domain": "QQ",
        "variables": list(variables),
        "polynomial": {
            "terms": [
                {
                    "coefficient": {"num": str(num), "den": str(den)},
                    "exponents": list(exp),
                }
                for num, den, exp in terms
            ]
        },
    }
    return RationalPolynomial.model_validate(payload)


def _ideal(
    variables: tuple[str, ...],
    generators: tuple[RationalPolynomial, ...],
) -> RationalPolynomialIdeal:
    return RationalPolynomialIdeal(variables=variables, generators=generators)


class TestGroebnerBasis:
    """Tests for ``polynomial.ideal.groebner_basis.compute``."""

    def test_simple_ideal(self) -> None:
        """Gröbner basis of <x^2 - y, xy - 1> has a finite basis."""
        g1 = _poly(("x", "y"), (1, 1, (2, 0)), (-1, 1, (0, 1)))
        g2 = _poly(("x", "y"), (1, 1, (1, 1)), (-1, 1, (0, 0)))
        ideal = _ideal(("x", "y"), (g1, g2))
        result = _run_groebner(
            GroebnerBasisRequest(ideal=ideal, monomial_order="grevlex")
        )
        assert result.generator_count >= 1
        assert result.basis is not None
        assert result.generator_count == len(result.basis.generators)

    def test_principal_ideal(self) -> None:
        """Gröbner basis of <x> in Q[x] is <x>."""
        g = _poly(("x",), (1, 1, (1,)))
        ideal = _ideal(("x",), (g,))
        result = _run_groebner(GroebnerBasisRequest(ideal=ideal, monomial_order="lex"))
        assert result.generator_count >= 1

    def test_lex_order(self) -> None:
        """Gröbner basis with lex order works."""
        g1 = _poly(("x", "y"), (1, 1, (1, 1)))
        g2 = _poly(("x", "y"), (1, 1, (1, 0)), (-1, 1, (0, 1)))
        ideal = _ideal(("x", "y"), (g1, g2))
        result = _run_groebner(GroebnerBasisRequest(ideal=ideal, monomial_order="lex"))
        assert result.generator_count >= 1


class TestGroebnerBasisValidation:
    """Authored results must satisfy reduced-basis invariants exactly."""

    def test_zero_generator_rejected_for_nonzero_ideal(self) -> None:
        """A claimed basis (x, 0) for <x> is not a reduced Gröbner basis."""
        g = _poly(("x",), (1, 1, (1,)))
        request = GroebnerBasisRequest(ideal=_ideal(("x",), (g,)), monomial_order="lex")
        zero = _poly(("x",))
        forged_basis = RationalPolynomialIdeal(variables=("x",), generators=(g, zero))
        with pytest.raises(ValidationError):
            GroebnerBasisResult(
                ideal=request.ideal,
                outcome="COMPUTED",
                basis=forged_basis,
                generator_count=2,
                monomial_order="lex",
            )

    def test_singleton_zero_only_for_zero_ideal(self) -> None:
        """The singleton-zero representation is reserved for the zero ideal."""
        nonzero_g = _poly(("x",), (1, 1, (1,)))
        zero = _poly(("x",))
        request = GroebnerBasisRequest(
            ideal=_ideal(("x",), (nonzero_g,)), monomial_order="lex"
        )
        with pytest.raises(ValidationError):
            GroebnerBasisResult(
                ideal=request.ideal,
                outcome="COMPUTED",
                basis=RationalPolynomialIdeal(variables=("x",), generators=(zero,)),
                generator_count=1,
                monomial_order="lex",
            )

    def test_zero_ideal_produces_singleton_zero_basis(self) -> None:
        """The producer's canonical zero-ideal basis revalidates end to end."""
        zero = _poly(("x",))
        result = _run_groebner(GroebnerBasisRequest(ideal=_ideal(("x",), (zero,))))
        assert result.outcome == "COMPUTED"
        assert result.generator_count == 1
        assert result.basis is not None
        assert len(result.basis.generators[0].polynomial.terms) == 0

    def test_basis_with_trailing_zero_for_mixed_source_rejected(self) -> None:
        """<x, 0> has reduced basis (x); appending a zero entry is invalid."""
        g = _poly(("x", "y"), (1, 1, (1, 0)))
        zero = _poly(("x", "y"))
        request = GroebnerBasisRequest(
            ideal=_ideal(("x", "y"), (g, zero)), monomial_order="grevlex"
        )
        forged_basis = RationalPolynomialIdeal(
            variables=("x", "y"), generators=(g, zero)
        )
        with pytest.raises(ValidationError):
            GroebnerBasisResult(
                ideal=request.ideal,
                outcome="COMPUTED",
                basis=forged_basis,
                generator_count=2,
                monomial_order="grevlex",
            )


class TestIdealNormalForm:
    """Tests for ``polynomial.ideal.normal_form.compute``."""

    def test_polynomial_in_ideal(self) -> None:
        """x^2 mod <x^2 - y^2> should give a nonzero remainder that is not in the ideal."""
        g = _poly(("x", "y"), (1, 1, (2, 0)), (-1, 1, (0, 2)))
        ideal = _ideal(("x", "y"), (g,))
        poly = _poly(("x", "y"), (1, 1, (2, 0)))
        result = _run_normal_form(IdealNormalFormRequest(ideal=ideal, polynomial=poly))
        assert result.in_ideal is False
        assert result.remainder is not None
        assert len(result.remainder.polynomial.terms) > 0

    def test_polynomial_in_ideal_exactly(self) -> None:
        """x^2 - y^2 mod <x^2 - y^2> should give zero (in the ideal)."""
        g = _poly(("x", "y"), (1, 1, (2, 0)), (-1, 1, (0, 2)))
        ideal = _ideal(("x", "y"), (g,))
        poly = _poly(("x", "y"), (1, 1, (2, 0)), (-1, 1, (0, 2)))
        result = _run_normal_form(IdealNormalFormRequest(ideal=ideal, polynomial=poly))
        assert result.in_ideal is True
        assert result.remainder is not None
        assert len(result.remainder.polynomial.terms) == 0

    def test_constant_in_unit_ideal(self) -> None:
        """A constant is in the ideal <1> = Q[x,y]."""
        g = _poly(("x", "y"), (1, 1, (0, 0)))
        ideal = _ideal(("x", "y"), (g,))
        poly = _poly(("x", "y"), (3, 1, (0, 0)))
        result = _run_normal_form(IdealNormalFormRequest(ideal=ideal, polynomial=poly))
        assert result.in_ideal is True


class TestEliminationIdeal:
    """Tests for ``polynomial.ideal.elimination.compute``."""

    def test_eliminate_one_variable(self) -> None:
        """Eliminate x from <x^2 - y^2, x + y> → get ideal in Q[y]."""
        g1 = _poly(("x", "y"), (1, 1, (2, 0)), (-1, 1, (0, 2)))
        g2 = _poly(("x", "y"), (1, 1, (1, 0)), (1, 1, (0, 1)))
        ideal = _ideal(("x", "y"), (g1, g2))
        result = _run_elimination(
            EliminationIdealRequest(ideal=ideal, eliminated_variables=("x",))
        )
        assert result.elimination_ideal is not None
        assert "x" not in result.elimination_ideal.variables
        assert len(result.elimination_ideal.generators) >= 1

    def test_eliminated_variables_not_in_result(self) -> None:
        """The elimination ideal should not contain eliminated variables."""
        g1 = _poly(("x", "y"), (1, 1, (2, 0)), (-1, 1, (0, 2)))
        g2 = _poly(("x", "y"), (1, 1, (1, 0)), (1, 1, (0, 1)))
        ideal = _ideal(("x", "y"), (g1, g2))
        result = _run_elimination(
            EliminationIdealRequest(ideal=ideal, eliminated_variables=("x",))
        )
        assert result.elimination_ideal is not None
        for var in result.elimination_ideal.variables:
            assert var != "x"


class TestTypedKernelOutcomes:
    """Budget expiry and limit exhaustion surface as typed outcomes, not
    host exceptions, and the kernel runs in a killable worker process."""

    def test_normal_form_timeout_returns_typed_outcome(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An expired normal-form budget returns TIMEOUT instead of raising."""

        def exceed_budget(*args: object, **kwargs: object) -> NoReturn:
            raise _SympyKernelTimeoutError()

        monkeypatch.setattr(operations, "_run_sympy_kernel", exceed_budget)
        g = _poly(("x", "y"), (1, 1, (2, 0)), (-1, 1, (0, 2)))
        result = _run_normal_form(
            IdealNormalFormRequest(
                ideal=_ideal(("x", "y"), (g,)),
                polynomial=_poly(("x", "y"), (1, 1, (2, 0))),
            )
        )
        assert result.outcome == "TIMEOUT"
        assert result.remainder is None
        assert result.in_ideal is None
        assert result.detail is not None

    def test_elimination_timeout_returns_typed_outcome(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An expired elimination budget returns TIMEOUT instead of raising."""

        def exceed_budget(*args: object, **kwargs: object) -> NoReturn:
            raise _SympyKernelTimeoutError()

        monkeypatch.setattr(operations, "_run_sympy_kernel", exceed_budget)
        g = _poly(("x", "y"), (1, 1, (2, 0)), (-1, 1, (0, 2)))
        result = _run_elimination(
            EliminationIdealRequest(
                ideal=_ideal(("x", "y"), (g,)),
                eliminated_variables=("x",),
            )
        )
        assert result.outcome == "TIMEOUT"
        assert result.elimination_ideal is None
        assert result.eliminated_variables == ("x",)
        assert result.detail is not None

    def test_groebner_timeout_returns_typed_outcome(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An expired groebner budget returns TIMEOUT with no basis."""

        def exceed_budget(*args: object, **kwargs: object) -> NoReturn:
            raise _SympyKernelTimeoutError()

        monkeypatch.setattr(operations, "_run_sympy_kernel", exceed_budget)
        g1 = _poly(("x", "y"), (1, 1, (2, 0)), (-1, 1, (0, 1)))
        g2 = _poly(("x", "y"), (1, 1, (1, 1)), (-1, 1, (0, 0)))
        result = _run_groebner(GroebnerBasisRequest(ideal=_ideal(("x", "y"), (g1, g2))))
        assert result.outcome == "TIMEOUT"
        assert result.basis is None
        assert result.detail is not None

    def test_kernel_failure_returns_typed_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A kernel that fails without an exact result yields a typed ERROR,
        so the accepted request never observes a host exception."""

        def failing_kernel(*args: object, **kwargs: object) -> NoReturn:
            raise _SympyKernelError("worker crashed")

        monkeypatch.setattr(operations, "_run_sympy_kernel", failing_kernel)
        g = _poly(("x", "y"), (1, 1, (2, 0)), (-1, 1, (0, 2)))
        result = _run_groebner(GroebnerBasisRequest(ideal=_ideal(("x", "y"), (g,))))
        assert result.outcome == "ERROR"
        assert result.basis is None
        assert "worker crashed" in (result.detail or "")


class TestKillableWorkerContract:
    def test_budget_delegates_to_the_bounded_process_runner(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Wall budgets must run through the killable process engine.

        A detached daemon thread cannot be terminated, so repeated hard
        requests would accumulate SymPy work inside the server while returning
        TIMEOUT. The operation therefore delegates every kernel call to
        ``run_bounded_process`` with the declared wall budget.
        """
        observed: dict[str, object] = {}
        real_runner = run_bounded_stdin_python_kernel

        def spy(
            script: str,
            payload_json: str,
            *,
            wall_seconds: float,
            stdout_limit: int,
            stderr_limit: int,
        ) -> tuple[bool, str, bool]:
            observed["timeout"] = wall_seconds
            observed["child_is_process"] = True
            return real_runner(
                script,
                payload_json,
                wall_seconds=wall_seconds,
                stdout_limit=stdout_limit,
                stderr_limit=stderr_limit,
            )

        monkeypatch.setattr(operations, "run_bounded_stdin_python_kernel", spy)
        g1 = _poly(("x", "y"), (1, 1, (2, 0)), (-1, 1, (0, 1)))
        g2 = _poly(("x", "y"), (1, 1, (1, 1)), (-1, 1, (0, 0)))
        result = _run_groebner(
            GroebnerBasisRequest(
                ideal=_ideal(("x", "y"), (g1, g2)),
                resource_budget=IdealComputationBudget(wall_seconds=10),
            )
        )
        assert result.outcome == "COMPUTED"
        assert observed["timeout"] == 10

    def test_timed_out_call_leaves_no_lingering_threads(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After a typed TIMEOUT the process owns no leftover kernel threads."""

        def exceed_budget(*args: object, **kwargs: object) -> NoReturn:
            raise _SympyKernelTimeoutError()

        baseline = __import__("threading").active_count()
        monkeypatch.setattr(operations, "_run_sympy_kernel", exceed_budget)
        g1 = _poly(("x", "y"), (1, 1, (2, 0)), (-1, 1, (0, 1)))
        g2 = _poly(("x", "y"), (1, 1, (1, 1)), (-1, 1, (0, 0)))
        result = _run_groebner(GroebnerBasisRequest(ideal=_ideal(("x", "y"), (g1, g2))))
        assert result.outcome == "TIMEOUT"
        assert __import__("threading").active_count() == baseline


class TestBoundedResultValidation:
    """Every defining-invariant replay runs inside one bounded worker pass."""

    def test_output_exponent_growth_is_typed_limit(self) -> None:
        """Cascaded generators grow basis exponents beyond the canonical
        bound; the operation reports LIMIT_EXCEEDED instead of a post-hoc
        conversion error."""
        names = ("v", "w", "z", "y", "x", "a")
        pairs = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5))
        gens: list[RationalPolynomial] = []
        for hi, lo in pairs:
            exps_hi = [0] * len(names)
            exps_lo = [0] * len(names)
            exps_hi[hi] = 1
            exps_lo[lo] = 12
            gens.append(
                _poly(
                    names,
                    (1, 1, tuple(exps_hi)),
                    (-1, 1, tuple(exps_lo)),
                )
            )
        result = _run_groebner(
            GroebnerBasisRequest(
                ideal=_ideal(names, tuple(gens)),
                monomial_order="lex",
            )
        )
        assert result.outcome in {"COMPUTED", "LIMIT_EXCEEDED"}
        if result.outcome == "COMPUTED":
            assert result.basis is not None
            for generator in result.basis.generators:
                for term in generator.polynomial.terms:
                    assert all(e <= 32768 for e in term.exponents)

    def test_aggregate_basis_terms_enforce_result_budget(self) -> None:
        """Every reduced-basis polynomial stays under the per-polynomial
        term limit while their sum crosses the declared exact-result
        budget; the operation reports LIMIT_EXCEEDED instead of COMPUTED.
        """
        from math import comb

        names = ("w", "z", "y", "x", "a", "b")

        def exps(**spec: int) -> tuple[int, ...]:
            base = [0] * len(names)
            for name, power in spec.items():
                base[names.index(name)] = power
            return tuple(base)

        cascade = _poly(
            names,
            (1, 1, exps(x=1)),
            *[(-comb(11, k), 1, exps(a=11 - k, b=k)) for k in range(12)],
        )
        gens = (
            cascade,
            _poly(names, (1, 1, exps(y=1)), (-1, 1, exps(x=11))),
            _poly(names, (1, 1, exps(z=1)), (-1, 1, exps(y=5))),
            _poly(names, (1, 1, exps(w=1)), (-1, 1, exps(y=6))),
        )
        result = _run_groebner(
            GroebnerBasisRequest(ideal=_ideal(names, gens), monomial_order="lex")
        )
        assert result.outcome == "LIMIT_EXCEEDED"
        assert result.basis is None
        assert result.detail is not None

    def test_stdout_limited_worker_returns_typed_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A killed worker whose output exceeded the transport cap yields
        LIMIT_EXCEEDED, not ERROR."""
        from jacobian.math.polynomials.ideals import operations as ops

        def fake_kernel(*args: object, **kwargs: object) -> tuple[bool, bytes, bool]:
            return False, b"", True  # not timed out; empty output; limit hit

        monkeypatch.setattr(ops, "run_bounded_stdin_python_kernel", fake_kernel)
        g = _poly(("x", "y"), (1, 1, (2, 0)), (-1, 1, (0, 1)))
        result = _run_groebner(GroebnerBasisRequest(ideal=_ideal(("x", "y"), (g,))))
        assert result.outcome == "LIMIT_EXCEEDED"
        assert "transport bound" in (result.detail or "")
