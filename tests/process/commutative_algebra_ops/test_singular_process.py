"""Failure-mode tests for the one-shot Singular process boundary."""

from __future__ import annotations

import os
import sys
import tempfile
import threading
from pathlib import Path

import pytest

from jacobian._exact import CanonicalRational
from jacobian.math.commutative_algebra_ops._models import IdealComputationBudget
from jacobian.math.commutative_algebra_ops._singular import (
    run_singular_ideal_operation,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialIdeal,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)
from jacobian.process import bounded_process_cancellation


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
        "jacobian.math._singular.shutil.which",
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


def test_cancellation_is_preserved_as_its_own_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _executable(tmp_path, "import time; time.sleep(30)")
    _select_executable(monkeypatch, executable)
    cancellation = threading.Event()
    cancellation.set()

    with bounded_process_cancellation(cancellation):
        result = run_singular_ideal_operation(
            "radical",
            _ideal(),
            None,
            IdealComputationBudget(),
        )

    assert result.outcome == "CANCELLED"
    assert result.ideal is None


def test_missing_backend_is_a_typed_unavailable_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "jacobian.math._singular.shutil.which",
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
        "jacobian.math._singular.tempfile.TemporaryDirectory",
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
        "jacobian.math._singular.shutil.which",
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


def test_unsupported_backend_version_is_typed_unavailability(
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

    assert result.outcome == "UNAVAILABLE"
    assert result.ideal is None
    assert result.detail == "The installed Singular release is unsupported."


def test_invocation_is_hermetic_and_version_precedes_algebra(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = (
        "JACOBIAN_SINGULAR_IDEAL_V1",
        "44105",
        "1",
        "GENERATOR",
        "1|2",
        "END_GENERATOR",
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

    result = run_singular_ideal_operation(
        "radical",
        _ideal(),
        None,
        IdealComputationBudget(),
    )

    assert result.outcome == "COMPUTED"
    source = (
        __import__(
            "jacobian.math.commutative_algebra_ops._singular",
            fromlist=["_script"],
        )
        ._script("radical", _ideal(), None)
        .decode("ascii")
    )
    assert source.index('system("version")') < source.index('LIB "primdec.lib"')


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
        "jacobian.math._singular.tempfile.TemporaryDirectory",
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
