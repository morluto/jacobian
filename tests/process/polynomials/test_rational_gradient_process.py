"""Process-boundary tests for rational-gradient recognition and cancellation."""

from __future__ import annotations

import json
from time import monotonic
from typing import Any

import pytest
from sympy import symbols

from jacobian import process
from jacobian._execution import (
    OperationExecutionTimeoutError,
    bind_request_deadline,
    request_execution,
)
from jacobian.math.polynomials._conversions import rational_function_from_sympy
from jacobian.math.polynomials.rational_functions.gradient import (
    _gcd_process,
    gradient,
)
from jacobian.math.polynomials.values import RationalFunction
from jacobian.process import BoundedProcessResult, ProcessResourceLimits


def _general_source() -> RationalFunction:
    x, y = symbols("x y")
    return rational_function_from_sympy((x * x + y) / (x - y), ("x", "y"))


def test_recognition_worker_timeout_uses_remaining_request_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}

    def timed_out(*_args: Any, **kwargs: Any) -> BoundedProcessResult:
        observed.update(kwargs)
        return BoundedProcessResult(
            returncode=None,
            stdout=b"",
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=True,
        )

    monkeypatch.setattr(process, "run_bounded_process", timed_out)
    started = monotonic()
    with (
        request_execution(started),
        pytest.raises(
            OperationExecutionTimeoutError,
            match="during coprimality recognition",
        ),
    ):
        bind_request_deadline(started + 5.0)
        gradient(_general_source())

    assert 0 < observed["timeout_seconds"] <= 5.0
    resource_limits = observed["resource_limits"]
    assert isinstance(resource_limits, ProcessResourceLimits)
    assert resource_limits.cpu_seconds is not None
    assert resource_limits.address_space_bytes is not None
    assert str(observed["cwd"]).split("/")[-1].startswith("jacobian-lie-recognition-")


def test_cancellation_worker_timeout_uses_remaining_request_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}

    def timed_out(*_args: Any, **kwargs: Any) -> BoundedProcessResult:
        observed.update(kwargs)
        return BoundedProcessResult(
            returncode=None,
            stdout=b"",
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=True,
        )

    monkeypatch.setattr(_gcd_process, "run_bounded_process", timed_out)
    started = monotonic()
    with (
        request_execution(started),
        pytest.raises(
            OperationExecutionTimeoutError,
            match="during denominator-derivative gcd",
        ),
    ):
        bind_request_deadline(started + 8.0)
        gradient(_general_source())

    payload = json.loads(observed["input_bytes"])
    assert payload["task"] == "derivative_gcds"
    assert payload["variable_count"] == 2
    assert payload["axes"] == [0, 1]
    assert len(payload["terms"]) == 2

    assert 0 < observed["timeout_seconds"] <= 8.0
    resource_limits = observed["resource_limits"]
    assert isinstance(resource_limits, ProcessResourceLimits)
    assert resource_limits.cpu_seconds is not None
    assert resource_limits.address_space_bytes is not None
    assert (
        str(observed["cwd"])
        .split("/")[-1]
        .startswith("jacobian-rational-gradient-gcd-")
    )
