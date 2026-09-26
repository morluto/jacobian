"""Tests for Groebner basis, normal form, and elimination ideal operations."""

from __future__ import annotations

import json
import time
from typing import Any, Literal, NoReturn, TypedDict

import pytest
from pydantic import ValidationError

from jacobian._execution import OperationExecutionTimeoutError
from jacobian.math.polynomials.ideals import _sympy_process, operations
from jacobian.math.polynomials.ideals._models import (
    EliminationIdealRequest,
    EliminationIdealResult,
    GroebnerBasisRequest,
    GroebnerBasisResult,
    IdealComputationBudget,
    IdealNormalFormRequest,
    IdealNormalFormResult,
)
from jacobian.math.polynomials.ideals._sympy_process import (
    _ResultLimitExceededError,
    _SympyKernelCancelledError,
    _SympyKernelError,
    _SympyKernelTimeoutError,
)
from jacobian.math.polynomials.ideals.operations import (
    elimination_ideal,
    groebner_basis,
    ideal_normal_form,
    verify_groebner_basis,
    verify_ideal_normal_form,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialIdeal,
)
from jacobian.process import BoundedProcessResult, run_bounded_process


def _run_groebner(request: GroebnerBasisRequest) -> GroebnerBasisResult:
    return groebner_basis(
        request.ideal, request.monomial_order, resource_budget=request.resource_budget
    )


def _run_normal_form(request: IdealNormalFormRequest) -> IdealNormalFormResult:
    return ideal_normal_form(request.ideal, request.polynomial, request.monomial_order)


def _run_elimination(request: EliminationIdealRequest) -> EliminationIdealResult:
    return elimination_ideal(
        request.ideal,
        request.eliminated_variables,
        resource_budget=request.resource_budget,
    )


class _NativeRationalComponents(TypedDict):
    num: int
    den: int


class _NativePolynomialTerm(TypedDict):
    coefficient: _NativeRationalComponents
    exponents: list[int]


class _NativePolynomialTerms(TypedDict):
    terms: list[_NativePolynomialTerm]


class _RationalPolynomialPayload(TypedDict):
    domain: Literal["QQ"]
    variables: list[str]
    polynomial: _NativePolynomialTerms


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
                    "coefficient": {"num": num, "den": den},
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

    @pytest.mark.parametrize(
        ("ideal", "monomial_order"),
        (
            (
                _ideal(
                    ("x", "y"),
                    (
                        _poly(("x", "y"), (1, 1, (2, 0)), (-1, 1, (0, 1))),
                        _poly(("x", "y"), (1, 1, (1, 1)), (-1, 1, (0, 0))),
                    ),
                ),
                "grevlex",
            ),
            (_ideal(("x",), (_poly(("x",), (1, 1, (1,))),)), "lex"),
            (
                _ideal(
                    ("x", "y"),
                    (
                        _poly(("x", "y"), (1, 1, (1, 1))),
                        _poly(("x", "y"), (1, 1, (1, 0)), (-1, 1, (0, 1))),
                    ),
                ),
                "lex",
            ),
        ),
        ids=("general-grevlex", "principal-lex", "two-generator-lex"),
    )
    def test_basis_is_source_bound_for_each_order(
        self, ideal: RationalPolynomialIdeal, monomial_order: str
    ) -> None:
        result = _run_groebner(
            GroebnerBasisRequest(ideal=ideal, monomial_order=monomial_order)
        )
        assert result.basis is not None
        assert result.generator_count == len(result.basis.generators)
        assert verify_groebner_basis(result)


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
                basis=RationalPolynomialIdeal(variables=("x",), generators=(zero,)),
                generator_count=1,
                monomial_order="lex",
            )

    def test_zero_ideal_produces_singleton_zero_basis(self) -> None:
        """The producer's canonical zero-ideal basis revalidates end to end."""
        zero = _poly(("x",))
        result = _run_groebner(GroebnerBasisRequest(ideal=_ideal(("x",), (zero,))))
        assert result.generator_count == 1
        assert result.basis is not None
        assert len(result.basis.generators[0].polynomial.terms) == 0
        assert verify_groebner_basis(result)

    def test_serialized_basis_verifier_rejects_forged_basis(self) -> None:
        g = _poly(("x",), (1, 1, (1,)))
        result = _run_groebner(
            GroebnerBasisRequest(ideal=_ideal(("x",), (g,)), monomial_order="lex")
        )
        decoded = type(result).model_validate_json(result.model_dump_json())
        assert verify_groebner_basis(decoded)

        payload = decoded.model_dump(mode="json")
        payload["basis"]["generators"][0]["polynomial"]["terms"][0]["coefficient"][
            "num"
        ] = "2"
        forged = type(result).model_validate_json(json.dumps(payload))

        assert not verify_groebner_basis(forged)

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
        assert result.remainder == _poly(("x", "y"), (1, 1, (0, 2)))

    @pytest.mark.parametrize(
        ("ideal", "polynomial"),
        (
            (
                _ideal(
                    ("x", "y"),
                    (_poly(("x", "y"), (1, 1, (2, 0)), (-1, 1, (0, 2))),),
                ),
                _poly(("x", "y"), (1, 1, (2, 0)), (-1, 1, (0, 2))),
            ),
            (
                _ideal(("x", "y"), (_poly(("x", "y"), (1, 1, (0, 0))),)),
                _poly(("x", "y"), (3, 1, (0, 0))),
            ),
        ),
        ids=("generator-reduces-to-zero", "constant-mod-unit"),
    )
    def test_polynomial_in_ideal_exactly(
        self, ideal: RationalPolynomialIdeal, polynomial: RationalPolynomial
    ) -> None:
        result = _run_normal_form(
            IdealNormalFormRequest(ideal=ideal, polynomial=polynomial)
        )
        assert result.in_ideal is True
        assert result.remainder is not None
        assert len(result.remainder.polynomial.terms) == 0

    def test_serialized_normal_form_verifier_rejects_forged_remainder(self) -> None:
        g = _poly(("x",), (1, 1, (1,)))
        result = _run_normal_form(
            IdealNormalFormRequest(
                ideal=_ideal(("x",), (g,)),
                polynomial=g,
            )
        )
        decoded = type(result).model_validate_json(result.model_dump_json())
        assert verify_ideal_normal_form(decoded)

        payload = decoded.model_dump(mode="json")
        payload["remainder"] = _poly(("x",), (1, 1, (0,))).model_dump(mode="json")
        payload["in_ideal"] = False
        forged = type(result).model_validate_json(json.dumps(payload))

        assert not verify_ideal_normal_form(forged)


class TestEliminationIdeal:
    """Tests for ``polynomial.ideal.elimination.compute``."""


class TestKernelFailures:
    """Operational non-completion establishes no ideal result."""

    def test_normal_form_timeout_raises_execution_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An expired normal-form budget establishes no remainder."""

        def exceed_budget(*args: object, **kwargs: object) -> NoReturn:
            raise _SympyKernelTimeoutError()

        monkeypatch.setattr(operations, "_run_sympy_kernel", exceed_budget)
        g = _poly(("x", "y"), (1, 1, (2, 0)), (-1, 1, (0, 2)))
        with pytest.raises(OperationExecutionTimeoutError):
            _run_normal_form(
                IdealNormalFormRequest(
                    ideal=_ideal(("x", "y"), (g,)),
                    polynomial=_poly(("x", "y"), (1, 1, (2, 0))),
                )
            )

    def test_elimination_timeout_raises_execution_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An expired elimination budget establishes no ideal."""

        def exceed_budget(*args: object, **kwargs: object) -> NoReturn:
            raise _SympyKernelTimeoutError()

        monkeypatch.setattr(operations, "_run_sympy_kernel", exceed_budget)
        g = _poly(("x", "y"), (1, 1, (2, 0)), (-1, 1, (0, 2)))
        with pytest.raises(OperationExecutionTimeoutError):
            _run_elimination(
                EliminationIdealRequest(
                    ideal=_ideal(("x", "y"), (g,)),
                    eliminated_variables=("x",),
                )
            )

    def test_groebner_timeout_raises_execution_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An expired Groebner budget establishes no basis."""

        def exceed_budget(*args: object, **kwargs: object) -> NoReturn:
            raise _SympyKernelTimeoutError()

        monkeypatch.setattr(operations, "_run_sympy_kernel", exceed_budget)
        g1 = _poly(("x", "y"), (1, 1, (2, 0)), (-1, 1, (0, 1)))
        g2 = _poly(("x", "y"), (1, 1, (1, 1)), (-1, 1, (0, 0)))
        with pytest.raises(OperationExecutionTimeoutError):
            _run_groebner(GroebnerBasisRequest(ideal=_ideal(("x", "y"), (g1, g2))))

    def test_kernel_failure_returns_typed_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed kernel raises instead of constructing a basis result."""

        def failing_kernel(*args: object, **kwargs: object) -> NoReturn:
            raise _SympyKernelError("worker crashed")

        monkeypatch.setattr(operations, "_run_sympy_kernel", failing_kernel)
        g = _poly(("x", "y"), (1, 1, (2, 0)), (-1, 1, (0, 2)))
        with pytest.raises(RuntimeError, match="worker crashed"):
            _run_groebner(GroebnerBasisRequest(ideal=_ideal(("x", "y"), (g,))))

    def test_native_decoding_respects_the_absolute_deadline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A native call without an envelope still enforces the deadline.

        The worker returns a response, but result decoding and assembly push
        past the shared wall limit. With no request execution envelope the
        checkpoints cannot see the absolute deadline, so the decode path must
        enforce it explicitly rather than return a basis past the limit.
        """

        def late_kernel(*args: object, **kwargs: object) -> dict[str, object]:
            return {"generators": []}

        monkeypatch.setattr(
            operations, "_run_relation_kernel_before_deadline", late_kernel
        )
        g = _poly(("x", "y"), (1, 1, (2, 0)), (-1, 1, (0, 2)))
        with pytest.raises(OperationExecutionTimeoutError):
            groebner_basis(
                _ideal(("x", "y"), (g,)),
                _outer_deadline=time.monotonic() - 1,
            )

    @pytest.mark.parametrize(
        "failure",
        (
            _SympyKernelTimeoutError,
            _SympyKernelCancelledError,
            _SympyKernelError,
            _ResultLimitExceededError,
        ),
    )
    def test_verifier_propagates_kernel_noncompletion(
        self,
        monkeypatch: pytest.MonkeyPatch,
        failure: type[Exception],
    ) -> None:
        g = _poly(("x",), (1, 1, (1,)))
        result = _run_groebner(
            GroebnerBasisRequest(ideal=_ideal(("x",), (g,)), monomial_order="lex")
        )

        def failing_kernel(*args: object, **kwargs: object) -> NoReturn:
            raise failure("worker did not produce a result")

        monkeypatch.setattr(operations, "_run_sympy_kernel", failing_kernel)
        with pytest.raises(failure, match="worker did not produce a result"):
            verify_groebner_basis(result)


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
        observed: dict[str, Any] = {}
        real_runner = run_bounded_process

        def spy(
            *args: object,
            **kwargs: object,
        ) -> BoundedProcessResult:
            observed["timeout"] = kwargs["timeout_seconds"]
            observed["command"] = args[0]
            observed["child_is_process"] = True
            return real_runner(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(_sympy_process, "run_bounded_process", spy)
        g1 = _poly(("x", "y"), (1, 1, (2, 0)), (-1, 1, (0, 1)))
        g2 = _poly(("x", "y"), (1, 1, (1, 1)), (-1, 1, (0, 0)))
        result = _run_groebner(
            GroebnerBasisRequest(
                ideal=_ideal(("x", "y"), (g1, g2)),
                resource_budget=IdealComputationBudget(wall_seconds=10),
            )
        )
        assert observed["timeout"] is not None
        assert float(observed["timeout"]) <= 10
        assert "-I" in observed["command"]
        assert result.basis is not None

class TestBoundedResultConstruction:
    """Worker result-envelope failures remain operational failures."""

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
        try:
            result = _run_groebner(
                GroebnerBasisRequest(
                    ideal=_ideal(names, tuple(gens)),
                    monomial_order="lex",
                )
            )
        except RuntimeError:
            pass
        else:
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
        with pytest.raises(RuntimeError, match="exact-result limit"):
            _run_groebner(
                GroebnerBasisRequest(ideal=_ideal(names, gens), monomial_order="lex")
            )

    def test_stdout_limited_worker_returns_backend_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Channel exhaustion does not establish a mathematical result limit."""

        def fake_kernel(*args: object, **kwargs: object) -> BoundedProcessResult:
            return BoundedProcessResult(
                returncode=None,
                stdout=b"",
                stderr=b"",
                stdout_exceeded=True,
                stderr_exceeded=False,
                timed_out=False,
            )

        monkeypatch.setattr(_sympy_process, "run_bounded_process", fake_kernel)
        g = _poly(("x", "y"), (1, 1, (2, 0)), (-1, 1, (0, 1)))
        with pytest.raises(RuntimeError, match="result channel bound"):
            _run_groebner(GroebnerBasisRequest(ideal=_ideal(("x", "y"), (g,))))
