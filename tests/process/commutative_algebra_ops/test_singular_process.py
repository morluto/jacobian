"""Failure-mode tests for the one-shot Singular process boundary."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

from jacobian._exact import CanonicalRational
from jacobian.math.commutative_algebra_ops._models import IdealComputationBudget
from jacobian.math.commutative_algebra_ops._singular import (
    run_singular_ideal_operation,
    run_singular_minimal_primes,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialIdeal,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)
from jacobian.process import BoundedProcessResult, ProcessResourceLimits


def _ideal() -> RationalPolynomialIdeal:
    variables = ("x",)
    return RationalPolynomialIdeal(
        variables=variables,
        generators=(
            RationalPolynomial(
                variables=variables,
                polynomial=SparseRationalPolynomial(
                    terms=(
                        RationalPolynomialTerm(
                            coefficient=CanonicalRational(num="1", den="1"),
                            exponents=(2,),
                        ),
                    )
                ),
            ),
        ),
    )


def _executable(tmp_path: Path, body: str) -> str:
    path = tmp_path / "fake-singular"
    path.write_text(f"#!{sys.executable}\n{body}\n", encoding="utf-8")
    path.chmod(0o700)
    return os.fspath(path)


def _select_executable(monkeypatch: pytest.MonkeyPatch, executable: str) -> None:
    monkeypatch.setattr(
        "jacobian.math.commutative_algebra_ops._singular.shutil.which",
        lambda name: executable if name == "Singular" else None,
    )


def test_timeout_is_not_reported_as_a_mathematical_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _executable(tmp_path, "import time; time.sleep(30)")
    _select_executable(monkeypatch, executable)
    result = run_singular_ideal_operation(
        "radical",
        _ideal(),
        None,
        IdealComputationBudget(wall_seconds=1),
    )
    assert result.outcome == "TIMEOUT"
    assert result.ideal is None


def test_missing_backend_is_a_typed_unavailable_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "jacobian.math.commutative_algebra_ops._singular.shutil.which",
        lambda name: None,
    )

    result = run_singular_ideal_operation(
        "radical",
        _ideal(),
        None,
        IdealComputationBudget(),
    )

    assert result.outcome == "UNAVAILABLE"
    assert result.ideal is None


def test_minimal_prime_family_protocol_is_typed_and_canonically_ordered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = (
        "JACOBIAN_SINGULAR_IDEAL_V1",
        "44000",
        "2",
        "COMPONENT",
        "1",
        "GENERATOR",
        "1|1",
        "END_GENERATOR",
        "END_COMPONENT",
        "COMPONENT",
        "1",
        "GENERATOR",
        "1|2",
        "END_GENERATOR",
        "END_COMPONENT",
        "END",
    )
    executable = _executable(tmp_path, f"print({chr(10).join(records)!r})")
    _select_executable(monkeypatch, executable)

    result = run_singular_minimal_primes(_ideal(), IdealComputationBudget())

    assert result.outcome == "COMPUTED"
    assert result.components is not None
    assert tuple(
        component.model_dump_json() for component in result.components
    ) == tuple(sorted(component.model_dump_json() for component in result.components))


def test_narrowed_replay_allowance_reaches_the_bounded_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def bounded_process(
        command: object,
        *,
        timeout_seconds: float,
        resource_limits: ProcessResourceLimits | None = None,
        **kwargs: object,
    ) -> BoundedProcessResult:
        seen["timeout_seconds"] = timeout_seconds
        seen["cpu_seconds"] = (
            resource_limits.cpu_seconds if resource_limits is not None else None
        )
        return BoundedProcessResult(
            returncode=0,
            stdout=b"",
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=True,
            cancelled=False,
        )

    monkeypatch.setattr(
        "jacobian.math.commutative_algebra_ops._singular.run_bounded_process",
        bounded_process,
    )
    monkeypatch.setattr(
        "jacobian.math.commutative_algebra_ops._singular.shutil.which",
        lambda name: "/usr/bin/Singular" if name == "Singular" else None,
    )

    result = run_singular_minimal_primes(
        _ideal(), IdealComputationBudget(wall_seconds=60), wall_seconds=6.75
    )

    assert result.outcome == "TIMEOUT"
    assert seen["timeout_seconds"] == 6.75
    assert seen["cpu_seconds"] == 6


def test_exhausted_replay_allowance_times_out_without_launching_singular(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(command: object, **kwargs: object) -> BoundedProcessResult:
        raise AssertionError("Singular must not launch on an exhausted allowance")

    monkeypatch.setattr(
        "jacobian.math.commutative_algebra_ops._singular.run_bounded_process",
        forbidden,
    )

    result = run_singular_minimal_primes(
        _ideal(), IdealComputationBudget(), wall_seconds=-0.5
    )

    assert result.outcome == "TIMEOUT"
    assert result.components is None


def test_caller_cannot_narrow_the_exact_result_contract() -> None:
    with pytest.raises(ValueError, match="greater than or equal to 1024"):
        IdealComputationBudget(maximum_output_terms=1)


def test_temporary_directory_failure_is_a_typed_unavailable_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _executable(tmp_path, 'print("not reached")')
    _select_executable(monkeypatch, executable)

    def unavailable_directory(*args: object, **kwargs: object) -> None:
        raise OSError("temporary storage unavailable")

    monkeypatch.setattr(
        "jacobian.math.commutative_algebra_ops._singular.tempfile.TemporaryDirectory",
        unavailable_directory,
    )

    result = run_singular_ideal_operation(
        "radical", _ideal(), None, IdealComputationBudget()
    )

    assert result.outcome == "UNAVAILABLE"


def test_relative_path_backend_is_resolved_before_entering_worker_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = Path(_executable(tmp_path, 'print("not the protocol")'))
    monkeypatch.chdir(tmp_path)
    _select_executable(monkeypatch, executable.name)

    result = run_singular_ideal_operation(
        "radical",
        _ideal(),
        None,
        IdealComputationBudget(),
    )

    assert result.outcome == "ERROR"
    assert (
        result.detail == "Singular returned an invalid or unsupported result encoding."
    )


def test_relative_prlimit_path_is_resolved_before_entering_worker_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = Path(_executable(tmp_path, 'print("not the protocol")'))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "jacobian.math.commutative_algebra_ops._singular.shutil.which",
        lambda name: executable.name if name in {"Singular", "prlimit"} else None,
    )

    result = run_singular_ideal_operation(
        "radical", _ideal(), None, IdealComputationBudget()
    )

    assert result.outcome == "ERROR"


def test_nonzero_exit_is_a_typed_execution_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _executable(tmp_path, "raise SystemExit(7)")
    _select_executable(monkeypatch, executable)
    result = run_singular_ideal_operation(
        "radical",
        _ideal(),
        None,
        IdealComputationBudget(),
    )
    assert result.outcome == "ERROR"
    assert result.ideal is None
    assert result.detail == "Singular failed without producing an exact ideal."


def test_malformed_success_output_is_a_typed_execution_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _executable(tmp_path, 'print("not the protocol")')
    _select_executable(monkeypatch, executable)
    result = run_singular_ideal_operation(
        "radical",
        _ideal(),
        None,
        IdealComputationBudget(),
    )
    assert result.outcome == "ERROR"
    assert result.ideal is None
    assert (
        result.detail == "Singular returned an invalid or unsupported result encoding."
    )


def test_unsupported_backend_version_is_a_typed_execution_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _executable(
        tmp_path,
        "print('\\n'.join(("
        "'JACOBIAN_SINGULAR_IDEAL_V1', '45000', '1', 'GENERATOR', "
        "'1|2', 'END_GENERATOR', 'END'))) ",
    )
    _select_executable(monkeypatch, executable)

    result = run_singular_ideal_operation(
        "radical",
        _ideal(),
        None,
        IdealComputationBudget(),
    )

    assert result.outcome == "ERROR"
    assert result.ideal is None
    assert (
        result.detail == "Singular returned an invalid or unsupported result encoding."
    )


def test_exact_result_limit_is_not_reported_as_invalid_backend_encoding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        "JACOBIAN_SINGULAR_IDEAL_V1",
        "44000",
        "1",
        "GENERATOR",
        *(f"1|{exponent}" for exponent in range(1_025)),
        "END_GENERATOR",
        "END",
    ]
    executable = _executable(tmp_path, f"print({chr(10).join(records)!r})")
    _select_executable(monkeypatch, executable)

    result = run_singular_ideal_operation(
        "radical", _ideal(), None, IdealComputationBudget()
    )

    assert result.outcome == "LIMIT_EXCEEDED"
    assert (
        result.detail == "The exact Singular ideal exceeds the declared result bound."
    )


def test_stderr_on_zero_exit_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _executable(
        tmp_path,
        'import sys; print("warning", file=sys.stderr)',
    )
    _select_executable(monkeypatch, executable)
    result = run_singular_ideal_operation(
        "radical",
        _ideal(),
        None,
        IdealComputationBudget(),
    )
    assert result.outcome == "ERROR"
    assert result.ideal is None


def test_oversized_stdout_is_a_typed_result_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _executable(tmp_path, 'print("x" * 600_000)')
    _select_executable(monkeypatch, executable)

    result = run_singular_ideal_operation(
        "radical",
        _ideal(),
        None,
        IdealComputationBudget(),
    )

    assert result.outcome == "LIMIT_EXCEEDED"
    assert result.ideal is None
    assert (
        result.detail == "The exact Singular ideal exceeds the declared result bound."
    )


def test_request_scoped_directory_is_removed_after_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _executable(tmp_path, 'print("not the protocol")')
    _select_executable(monkeypatch, executable)
    created: list[Path] = []

    class RecordingTemporaryDirectory(tempfile.TemporaryDirectory[str]):
        def __enter__(self) -> str:
            directory = super().__enter__()
            created.append(Path(directory))
            return directory

    monkeypatch.setattr(
        "jacobian.math.commutative_algebra_ops._singular.tempfile.TemporaryDirectory",
        RecordingTemporaryDirectory,
    )

    run_singular_ideal_operation(
        "radical",
        _ideal(),
        None,
        IdealComputationBudget(),
    )

    assert created
    assert all(not directory.exists() for directory in created)
