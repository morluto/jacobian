"""Failure-mode tests for the generic-fiber Singular boundary."""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import pytest

from jacobian._exact import CanonicalRational
from jacobian.math import _singular as shared_singular
from jacobian.math.polynomials.maps import (
    RationalPolynomialMap,
    _operations,
    _singular,
)
from jacobian.math.polynomials.maps._models import (
    GenericDegreeComputationBudget,
    GenericDegreeRequest,
    GenericFiberCertificate,
    GenericFiberPolynomial,
    GenericFiberTerm,
)
from jacobian.math.polynomials.maps._operations import compute_generic_degree
from jacobian.math.polynomials.values import (
    RationalFunction,
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)
from jacobian.process import bounded_process_cancellation


def _map() -> RationalPolynomialMap:
    return RationalPolynomialMap(
        input_variables=("x",),
        output_polynomials=(
            RationalPolynomial(
                variables=("x",),
                polynomial=SparseRationalPolynomial(
                    terms=(
                        RationalPolynomialTerm(
                            coefficient=CanonicalRational(num="1", den="1"),
                            exponents=(1,),
                        ),
                    )
                ),
            ),
        ),
    )


def _two_variable_map() -> RationalPolynomialMap:
    variables = ("x", "y")
    return RationalPolynomialMap(
        input_variables=variables,
        output_polynomials=tuple(
            RationalPolynomial(
                variables=variables,
                polynomial=SparseRationalPolynomial(
                    terms=(
                        RationalPolynomialTerm(
                            coefficient=CanonicalRational(num="1", den="1"),
                            exponents=exponents,
                        ),
                    )
                ),
            )
            for exponents in ((1, 0), (0, 1))
        ),
    )


def _executable(tmp_path: Path, body: str) -> str:
    path = tmp_path / "fake-singular"
    path.write_text(f"#!{sys.executable}\n{body}\n", encoding="utf-8")
    path.chmod(0o700)
    return os.fspath(path)


def _select_executable(
    monkeypatch: pytest.MonkeyPatch,
    executable: str,
) -> None:
    monkeypatch.setattr(
        shared_singular.shutil,
        "which",
        lambda name: executable if name == "Singular" else None,
    )


def test_valid_protocol_is_replayed_to_a_mathematical_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = (
        "JACOBIAN_SINGULAR_GENERIC_FIBER_V1",
        "44105",
        "0",
        "1",
        "1",
        "1",
        "POLYNOMIAL",
        "1",
        "1",
        "1",
        "0",
        "(-jtp1)",
        "1",
        "END_POLYNOMIAL",
        "POLYNOMIAL",
        "0",
        "1",
        "1",
        "END_POLYNOMIAL",
        "END",
    )
    executable = _executable(tmp_path, f"print({chr(10).join(records)!r})")
    _select_executable(monkeypatch, executable)

    result = compute_generic_degree(GenericDegreeRequest(polynomial_map=_map()))

    assert result.outcome == "GENERICALLY_FINITE"
    assert result.degree == 1


def test_invocation_disables_ambient_startup_shell_and_standard_library(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = (
        "JACOBIAN_SINGULAR_GENERIC_FIBER_V1",
        "44105",
        "0",
        "1",
        "1",
        "1",
        "POLYNOMIAL",
        "1",
        "1",
        "1",
        "0",
        "(-jtp1)",
        "1",
        "END_POLYNOMIAL",
        "POLYNOMIAL",
        "0",
        "1",
        "1",
        "END_POLYNOMIAL",
        "END",
    )
    body = (
        "import sys\n"
        "required={'-q','-t','--no-rc','--no-shell','--no-stdlib'}\n"
        "if not required.issubset(sys.argv): raise SystemExit(7)\n"
        f"print({chr(10).join(records)!r})"
    )
    executable = _executable(tmp_path, body)
    _select_executable(monkeypatch, executable)

    result = compute_generic_degree(GenericDegreeRequest(polynomial_map=_map()))

    assert result.outcome == "GENERICALLY_FINITE"


def test_timeout_is_not_a_dominance_conclusion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _executable(tmp_path, "import time; time.sleep(30)")
    _select_executable(monkeypatch, executable)

    result = compute_generic_degree(
        GenericDegreeRequest(
            polynomial_map=_map(),
            resource_budget=GenericDegreeComputationBudget(wall_seconds=1),
        )
    )

    assert result.outcome == "TIMEOUT"
    assert result.degree is None
    assert result.evidence is None


def test_cancellation_is_preserved_as_its_own_public_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _executable(tmp_path, "import time; time.sleep(30)")
    _select_executable(monkeypatch, executable)
    cancellation = threading.Event()
    cancellation.set()

    with bounded_process_cancellation(cancellation):
        result = compute_generic_degree(GenericDegreeRequest(polynomial_map=_map()))

    assert result.outcome == "CANCELLED"
    assert result.degree is None
    assert result.evidence is None


def test_malformed_success_output_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _executable(tmp_path, 'print("not the protocol")')
    _select_executable(monkeypatch, executable)

    result = compute_generic_degree(GenericDegreeRequest(polynomial_map=_map()))

    assert result.outcome == "ERROR"
    assert result.degree is None


def test_oversized_certificate_is_a_typed_bound_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _executable(tmp_path, 'print("x" * 600_000)')
    _select_executable(monkeypatch, executable)

    result = compute_generic_degree(GenericDegreeRequest(polynomial_map=_map()))

    assert result.outcome == "BOUND_EXCEEDED"
    assert result.degree is None


def test_standard_monomial_candidates_are_distinct_from_returned_monomials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        "JACOBIAN_SINGULAR_GENERIC_FIBER_V1",
        "44105",
        "0",
        "127",
        "3",
        "2",
    ]
    for exponents in ("64,0", "1,1", "0,64"):
        records.extend(("POLYNOMIAL", exponents, "1", "1", "END_POLYNOMIAL"))
    for _ in range(6):
        records.extend(("POLYNOMIAL", "END_POLYNOMIAL"))
    records.append("END")
    executable = _executable(tmp_path, f"print({chr(10).join(records)!r})")
    _select_executable(monkeypatch, executable)

    backend = _singular.run_singular_generic_fiber(
        _two_variable_map(),
        GenericDegreeComputationBudget(),
    )

    assert backend.outcome == "COMPUTED"
    assert backend.vector_dimension == 127
    assert backend.certificate is not None
    assert len(backend.certificate.standard_monomials) == 127


def _stripe_certificate() -> GenericFiberCertificate:
    """A structurally admitted certificate whose replay is genuinely heavy."""

    coefficient = RationalFunction(
        variables=("t1", "t2"),
        numerator=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num="1", den="1"),
                    exponents=(0, 0),
                ),
            )
        ),
        denominator=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num="1", den="1"),
                    exponents=(0, 0),
                ),
            )
        ),
    )
    basis = []
    for stripe in range(16):
        constant = 256 * stripe + 300
        basis.append(
            GenericFiberPolynomial(
                terms=tuple(
                    GenericFiberTerm(
                        coefficient=coefficient,
                        source_exponents=(offset, constant - offset),
                    )
                    for offset in sorted(range(240), reverse=True)
                )
            )
        )
    unit = GenericFiberPolynomial(
        terms=(GenericFiberTerm(coefficient=coefficient, source_exponents=(0, 0)),)
    )
    empty = GenericFiberPolynomial()
    return GenericFiberCertificate(
        target_parameters=("t1", "t2"),
        source_variable_order=("x", "y"),
        basis=tuple(basis),
        basis_from_source=(
            tuple(unit if column % 2 == 0 else empty for column in range(16)),
            tuple(unit if column % 2 == 1 else empty for column in range(16)),
        ),
    )


def test_heavy_certificate_replay_is_killably_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _operations,
        "run_singular_generic_fiber",
        lambda *_args: _singular.SingularGenericFiberResult(
            outcome="COMPUTED",
            certificate=_stripe_certificate(),
            dimension=1,
            vector_dimension=None,
            backend_version="4.4.1",
        ),
    )

    result = compute_generic_degree(
        GenericDegreeRequest(
            polynomial_map=_two_variable_map(),
            resource_budget=GenericDegreeComputationBudget(wall_seconds=1),
        )
    )

    assert result.outcome == "TIMEOUT"
    assert result.detail in (
        "Certificate replay exceeded the declared wall-time limit.",
        "The declared wall-time envelope expired before certificate replay.",
    )
    assert result.degree is None
    assert result.evidence is None


def test_one_second_budget_still_replays_a_light_certificate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = (
        "JACOBIAN_SINGULAR_GENERIC_FIBER_V1",
        "44105",
        "0",
        "1",
        "1",
        "1",
        "POLYNOMIAL",
        "1",
        "1",
        "1",
        "0",
        "(-jtp1)",
        "1",
        "END_POLYNOMIAL",
        "POLYNOMIAL",
        "0",
        "1",
        "1",
        "END_POLYNOMIAL",
        "END",
    )
    executable = _executable(tmp_path, f"print({chr(10).join(records)!r})")
    _select_executable(monkeypatch, executable)

    result = compute_generic_degree(
        GenericDegreeRequest(
            polynomial_map=_map(),
            resource_budget=GenericDegreeComputationBudget(wall_seconds=1),
        )
    )

    assert result.outcome == "GENERICALLY_FINITE"
    assert result.degree == 1
    assert result.evidence is not None


def test_expired_deadline_after_the_backend_still_reports_pre_replay_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def slow_backend(*_args):
        time.sleep(1.5)
        return _singular.SingularGenericFiberResult(
            outcome="COMPUTED",
            certificate=_stripe_certificate(),
            dimension=0,
            vector_dimension=3,
            backend_version="4.4.1",
        )

    monkeypatch.setattr(_operations, "run_singular_generic_fiber", slow_backend)

    result = compute_generic_degree(
        GenericDegreeRequest(
            polynomial_map=_two_variable_map(),
            resource_budget=GenericDegreeComputationBudget(wall_seconds=1),
        )
    )

    assert result.outcome == "TIMEOUT"
    assert (
        result.detail
        == "The declared wall-time envelope expired before certificate replay."
    )
    assert result.degree is None
    assert result.evidence is None


@pytest.mark.parametrize(
    "text",
    (
        "jtp1+system(quit)",
        "jtp1/2",
        "jtp2",
        "jtp1**2",
        "__import__",
    ),
)
def test_parameter_protocol_is_non_evaluating_and_fail_closed(text: str) -> None:
    with pytest.raises(ValueError):
        _singular._parse_parameter_polynomial(text, parameter_count=1)


def test_backend_script_uses_only_fixed_internal_identifiers() -> None:
    caller_named_map = RationalPolynomialMap(
        input_variables=("callerVariable",),
        output_polynomials=(
            RationalPolynomial(
                variables=("callerVariable",),
                polynomial=_map().output_polynomials[0].polynomial,
            ),
        ),
    )

    source = _singular._generic_fiber_script(caller_named_map).decode("ascii")

    assert "callerVariable" not in source
    assert "jv1" in source
    assert "jtp1" in source
    assert source.index('system("version")') < source.index("ring jacobian_ring")
