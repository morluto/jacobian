"""Keep real radix worker deadlines distinct from process startup failures."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from jacobian import _execution, process
from jacobian._execution import (
    BackendFailureReason,
    OperationBackendError,
    OperationExecutionTimeoutError,
    request_execution,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.algebraic_numbers import (
    _radix_prefix_process as radix_process,
)
from jacobian.math.number_theory.algebraic_numbers import radix_prefix
from jacobian.math.number_theory.algebraic_numbers.real import RealAlgebraicValue
from jacobian.process import ProcessPlatformTools, ProcessResourceLimits


@pytest.mark.parametrize("phase", ["supervisor", "decode"])
def test_native_radix_worker_timeout_is_not_a_startup_failure(
    monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    started = time.monotonic()
    deadline = started + 30
    clock = [started]
    completed: list[process.BoundedProcessResult] = []
    children: list[subprocess.Popen[bytes]] = []
    run = process.run_bounded_process

    def capture(*args: Any, **kwargs: Any) -> process.BoundedProcessResult:
        result = run(*args, **kwargs)
        completed.append(result)
        if phase == "decode":
            # The real worker delivered its own successful result frame. The
            # request expires at the next checked decoding boundary.
            clock[0] = deadline + 1
        return result

    monkeypatch.setattr(process, "run_bounded_process", capture)
    if phase == "supervisor":
        apply_limits = process._apply_post_start_limits

        def finish_setup(
            child: subprocess.Popen[bytes],
            limits: ProcessResourceLimits | None,
            applied_before_exec: bool,
            tools: ProcessPlatformTools | None,
        ) -> None:
            apply_limits(child, limits, applied_before_exec, tools)
            children.append(child)
            # Consume the execution portion during real process setup while
            # retaining its finite cleanup allowance. No worker is replaced.
            clock[0] = deadline - 0.25

        monkeypatch.setattr(process, "_apply_post_start_limits", finish_setup)
        monkeypatch.setattr(
            process, "time", SimpleNamespace(monotonic=lambda: clock[0])
        )
    else:
        monkeypatch.setattr(
            _execution, "time", SimpleNamespace(monotonic=lambda: clock[0])
        )

    with (
        request_execution(started, outer_deadline=deadline),
        pytest.raises(OperationExecutionTimeoutError),
    ):
        radix_prefix(
            RealAlgebraicValue(polynomial=(1, 0, -2), real_root_index=1), 10, 2
        )

    assert len(completed) == 1
    result = completed[0]
    assert not result.cancelled
    if phase == "supervisor":
        assert children and children[0].poll() is not None
        assert result.timed_out
    else:
        assert result.returncode == 0
        assert not result.timed_out
        assert (
            result.stdout
            == b'{"kind":"result","result":{"tag":"success","scaled_floor":"141"}}\n'
        )


def test_native_radix_ordinary_startup_failure_remains_a_backend_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Let the real Popen fail at the operating-system boundary.
    monkeypatch.setattr(
        radix_process,
        "sys",
        SimpleNamespace(executable=str(tmp_path / "missing-python")),
    )
    with pytest.raises(OperationBackendError) as caught:
        radix_prefix(
            RealAlgebraicValue(polynomial=(1, 0, -2), real_root_index=1), 10, 2
        )
    assert caught.value.reason is BackendFailureReason.STARTUP
    assert isinstance(caught.value.__cause__, FileNotFoundError)


def test_native_radix_real_worker_success_control() -> None:
    result = radix_prefix(
        RealAlgebraicValue(polynomial=(1, 0, -2), real_root_index=1), 10, 2
    )
    assert result.integer_part == 1
    assert result.fractional_digits == (4, 1)


@pytest.mark.parametrize(
    ("polynomial", "root_index", "code"),
    [
        ((1, 0, -1), 0, "real_algebraic.not_irreducible"),
        ((1, 0, 0, -2), 1, "algebraic_number.radix_root_index"),
    ],
)
def test_native_radix_real_worker_domain_error_control(
    polynomial: tuple[int, ...], root_index: int, code: str
) -> None:
    with pytest.raises(OperationDomainValidationError) as caught:
        radix_prefix(
            RealAlgebraicValue(polynomial=polynomial, real_root_index=root_index), 10, 2
        )
    assert caught.value.errors()[0]["type"] == code
