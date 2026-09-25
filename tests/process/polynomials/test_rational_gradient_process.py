"""Process-boundary tests for rational-gradient recognition and cancellation."""

from __future__ import annotations

import json
from time import monotonic
from typing import Any, NoReturn

import pytest
from sympy import symbols

from jacobian import process
from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    bind_request_deadline,
    request_execution,
)
from jacobian.math.polynomials._conversions import rational_function_from_sympy
from jacobian.math.polynomials.rational_functions import RationalFunctionMap
from jacobian.math.polynomials.rational_functions.gradient import (
    gradient,
)
from jacobian.math.polynomials.rational_functions.maps import jacobian_matrix
from jacobian.math.polynomials.values import RationalFunction
from jacobian.process import BoundedProcessResult, ProcessResourceLimits


def _completed(
    *,
    returncode: int | None = 0,
    stdout: bytes = b"",
    stderr: bytes = b"",
    stdout_exceeded: bool = False,
    stderr_exceeded: bool = False,
    timed_out: bool = False,
    cancelled: bool = False,
) -> BoundedProcessResult:
    """Name the operational state instead of repeating result fields."""
    return BoundedProcessResult(
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        stdout_exceeded=stdout_exceeded,
        stderr_exceeded=stderr_exceeded,
        timed_out=timed_out,
        cancelled=cancelled,
    )


def _general_source() -> RationalFunction:
    x, y = symbols("x y")
    return rational_function_from_sympy((x * x + y) / (x - y), ("x", "y"))


def test_gradient_admission_worker_timeout_uses_remaining_request_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}

    def timed_out(*_args: Any, **kwargs: Any) -> BoundedProcessResult:
        observed.update(kwargs)
        return _completed(returncode=None, timed_out=True)

    monkeypatch.setattr(process, "run_bounded_process", timed_out)
    started = monotonic()
    with (
        request_execution(started),
        pytest.raises(
            OperationExecutionTimeoutError,
            match="gradient deadline expired",
        ),
    ):
        bind_request_deadline(started + 5.0)
        gradient(_general_source())

    assert 0 < observed["timeout_seconds"] <= 5.0
    resource_limits = observed["resource_limits"]
    assert isinstance(resource_limits, ProcessResourceLimits)
    assert resource_limits.cpu_seconds is not None
    assert resource_limits.address_space_bytes is not None
    assert json.loads(observed["input_bytes"])["task"] == "gradient_admission"
    assert str(observed["cwd"]).split("/")[-1].startswith(
        "jacobian-rational-gradient-gcd-"
    )


def test_general_gradient_keeps_quotient_expansion_out_of_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sympy import Poly

    source = _general_source()
    expected = gradient(source)

    def parent_expansion_forbidden(*_args: Any, **_kwargs: Any) -> NoReturn:
        raise AssertionError("quotient expansion ran in the parent")

    monkeypatch.setattr(Poly, "diff", parent_expansion_forbidden)
    monkeypatch.setattr(Poly, "__mul__", parent_expansion_forbidden)
    assert gradient(source) == expected


def test_jacobian_rows_keep_quotient_expansion_out_of_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sympy import Poly

    component = _general_source()
    source = RationalFunctionMap(
        source_variables=("x", "y"),
        target_coordinates=("u",),
        components=(component,),
    )
    expected = jacobian_matrix(source)

    def parent_expansion_forbidden(*_args: Any, **_kwargs: Any) -> NoReturn:
        raise AssertionError("quotient expansion ran in the parent")

    monkeypatch.setattr(Poly, "diff", parent_expansion_forbidden)
    monkeypatch.setattr(Poly, "__mul__", parent_expansion_forbidden)
    assert jacobian_matrix(source) == expected


def test_cancellation_worker_timeout_uses_remaining_request_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian import process

    observed: dict[str, Any] = {}

    original = process.run_bounded_process

    def timed_out(*args: Any, **kwargs: Any) -> BoundedProcessResult:
        payload = json.loads(kwargs["input_bytes"])
        if payload.get("task") != "differentiate_batch":
            return original(*args, **kwargs)
        observed.update(kwargs)
        return _completed(returncode=None, timed_out=True)

    monkeypatch.setattr(process, "run_bounded_process", timed_out)
    started = monotonic()
    with (
        request_execution(started),
        pytest.raises(
            OperationExecutionTimeoutError,
            match="during gradient kernel",
        ),
    ):
        bind_request_deadline(started + 8.0)
        gradient(_general_source())

    payload = json.loads(observed["input_bytes"])
    assert payload["task"] == "differentiate_batch"
    assert [item["axis"] for item in payload["derivatives"]] == [0, 1]
    assert payload["variable_count"] == 2
    assert len(payload["numerator"]) == 2
    assert len(payload["denominator"]) == 2

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


def test_general_gradient_batches_active_axes_in_one_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[dict[str, Any]] = []
    original = process.run_bounded_process

    def record(*args: Any, **kwargs: Any) -> BoundedProcessResult:
        payload = json.loads(kwargs["input_bytes"])
        if payload.get("task") == "differentiate_batch":
            observed.append(payload)
        return original(*args, **kwargs)

    monkeypatch.setattr(process, "run_bounded_process", record)
    gradient(_general_source())

    assert len(observed) == 1
    assert [item["axis"] for item in observed[0]["derivatives"]] == [0, 1]


def test_gradient_kernel_cancellation_is_typed_non_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian import process

    original = process.run_bounded_process

    def cancelled(*args: Any, **kwargs: Any) -> BoundedProcessResult:
        payload = json.loads(kwargs["input_bytes"])
        if payload.get("task") != "differentiate_batch":
            return original(*args, **kwargs)
        return _completed(returncode=None, cancelled=True)

    monkeypatch.setattr(process, "run_bounded_process", cancelled)
    started = monotonic()
    with (
        request_execution(started),
        pytest.raises(
            OperationExecutionCancelledError,
            match="cancelled during gradient kernel",
        ),
    ):
        bind_request_deadline(started + 60.0)
        gradient(_general_source())
