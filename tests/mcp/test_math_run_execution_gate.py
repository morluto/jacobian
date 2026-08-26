"""Owner-declared MCP execution admission, cancellation, and capacity contracts."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Any, cast

import pytest
from mcp.server.mcpserver.exceptions import ToolError
from mcp.shared.exceptions import MCPError

from jacobian._models import StrictModel
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import MathTool, OperationResult
from jacobian.mcp.runtime import AppState
from jacobian.mcp.tools import math_run


class _Request(StrictModel):
    value: int


class _Result(StrictModel):
    value: int


def _context(state: AppState, cancellation: threading.Event) -> Any:
    return cast(
        Any,
        SimpleNamespace(
            request_context=SimpleNamespace(
                lifespan_context=state,
                session=SimpleNamespace(
                    _request_outbound=SimpleNamespace(cancel_requested=cancellation)
                ),
            )
        ),
    )


def _call(
    state: AppState,
    operation_id: str,
    value: int,
    cancellation: threading.Event,
    outcomes: dict[int, OperationResult | Exception],
) -> None:
    try:
        outcomes[value] = math_run(
            operation_id,
            {"value": value},
            ctx=_context(state, cancellation),
        )
    except Exception as exc:
        outcomes[value] = exc


def test_builtin_concurrency_is_an_explicit_owner_declaration() -> None:
    catalog = Catalog.open()

    for operation_id in ("sat.cnf.canonicalize", "sat.assignment.check"):
        operation = catalog.operation(operation_id)
        assert operation is not None
        assert operation.execution_admission.permits_concurrency

    unverified = catalog.operation("integer.compute.extended_gcd")
    assert unverified is not None
    assert not unverified.execution_admission.permits_concurrency


def test_unverified_owner_path_is_serialized_under_contention() -> None:
    active_calls = 0
    maximum_active_calls = 0
    active_lock = threading.Lock()

    def kernel(request: _Request) -> _Result:
        nonlocal active_calls, maximum_active_calls
        with active_lock:
            active_calls += 1
            maximum_active_calls = max(maximum_active_calls, active_calls)
        try:
            time.sleep(0.005)
            return _Result(value=request.value)
        finally:
            with active_lock:
                active_calls -= 1

    tool = MathTool(
        operation_id="test.execution_admission.serialized",
        title="Serialized owner path",
        description="Detects overlapping entry into an unverified kernel.",
        request_type=_Request,
        result_type=_Result,
        run=kernel,
    )
    state = AppState(operation_catalog=Catalog((tool,)))
    worker_count = 24
    start = threading.Barrier(worker_count + 1)
    outcomes: dict[int, OperationResult | Exception] = {}

    def worker(value: int) -> None:
        start.wait(timeout=2)
        _call(state, tool.operation_id, value, threading.Event(), outcomes)

    workers = [threading.Thread(target=worker, args=(value,)) for value in range(24)]
    for worker_thread in workers:
        worker_thread.start()
    start.wait(timeout=2)
    for worker_thread in workers:
        worker_thread.join(timeout=5)
        assert not worker_thread.is_alive()

    assert maximum_active_calls == 1
    assert len(outcomes) == worker_count
    assert all(isinstance(outcome, OperationResult) for outcome in outcomes.values())


def test_math_run_rejects_cancelled_request_while_waiting_for_owner_gate() -> None:
    invocations: list[int] = []
    first_started = threading.Event()
    release_first = threading.Event()

    def kernel(request: _Request) -> _Result:
        invocations.append(request.value)
        if request.value == 1:
            first_started.set()
            assert release_first.wait(timeout=2)
        return _Result(value=request.value)

    tool = MathTool(
        operation_id="test.execution_admission.cancelled",
        title="Cancelled owner gate",
        description="Records whether a cancelled waiter enters the kernel.",
        request_type=_Request,
        result_type=_Result,
        run=kernel,
    )
    state = AppState(operation_catalog=Catalog((tool,)))
    outcomes: dict[int, OperationResult | Exception] = {}
    first = threading.Thread(
        target=_call,
        args=(state, tool.operation_id, 1, threading.Event(), outcomes),
    )
    cancellation = threading.Event()
    second = threading.Thread(
        target=_call,
        args=(state, tool.operation_id, 2, cancellation, outcomes),
    )

    first.start()
    assert first_started.wait(timeout=1)
    second.start()
    time.sleep(0.1)
    cancellation.set()
    second.join(timeout=1)
    assert not second.is_alive()
    release_first.set()
    first.join(timeout=1)
    assert not first.is_alive()

    assert isinstance(outcomes[2], ToolError)
    assert "cancelled before execution" in str(outcomes[2])
    assert invocations == [1]


def test_math_run_validates_before_owner_capacity_admission() -> None:
    first_started = threading.Event()
    release_first = threading.Event()

    def kernel(request: _Request) -> _Result:
        first_started.set()
        assert release_first.wait(timeout=2)
        return _Result(value=request.value)

    tool = MathTool(
        operation_id="test.execution_admission.validation",
        title="Owner admission validation",
        description="Validates before waiting for execution capacity.",
        request_type=_Request,
        result_type=_Result,
        run=kernel,
    )
    state = AppState(
        operation_catalog=Catalog((tool,)),
        execution_queue_wait_seconds=0.05,
    )
    outcomes: dict[int, OperationResult | Exception] = {}
    holder = threading.Thread(
        target=_call,
        args=(state, tool.operation_id, 1, threading.Event(), outcomes),
    )
    holder.start()
    assert first_started.wait(timeout=1)
    try:
        with pytest.raises(MCPError) as invalid_error:
            math_run(
                tool.operation_id,
                {"value": "invalid"},
                ctx=_context(state, threading.Event()),
            )
        assert invalid_error.value.code == -32602
        assert invalid_error.value.data["code"] == "INVALID_REQUEST"
    finally:
        release_first.set()
        holder.join(timeout=1)
    assert not holder.is_alive()


def test_math_run_reports_typed_server_busy_after_bounded_wait() -> None:
    first_started = threading.Event()
    release_first = threading.Event()

    def kernel(request: _Request) -> _Result:
        first_started.set()
        assert release_first.wait(timeout=2)
        return _Result(value=request.value)

    tool = MathTool(
        operation_id="test.execution_admission.busy",
        title="Busy owner gate",
        description="Exercises bounded MCP capacity admission.",
        request_type=_Request,
        result_type=_Result,
        run=kernel,
    )
    state = AppState(
        operation_catalog=Catalog((tool,)),
        execution_queue_wait_seconds=0.05,
    )
    outcomes: dict[int, OperationResult | Exception] = {}
    holder = threading.Thread(
        target=_call,
        args=(state, tool.operation_id, 1, threading.Event(), outcomes),
    )
    holder.start()
    assert first_started.wait(timeout=1)
    try:
        with pytest.raises(MCPError) as busy_error:
            math_run(
                tool.operation_id,
                {"value": 2},
                ctx=_context(state, threading.Event()),
            )
        assert busy_error.value.code == -32_001
        assert busy_error.value.message == "mathematical execution capacity is busy"
        busy_data = busy_error.value.data
        assert busy_data["code"] == "SERVER_BUSY"
        assert busy_data["stage"] == "execution_admission"
        assert busy_data["operation_id"] == tool.operation_id
        assert busy_data["queue_wait_ms"] >= 50
        assert busy_data["queue_wait_limit_ms"] == 50
        assert busy_data["retryable"] is True
        assert "Retry" in busy_data["hint"]
    finally:
        release_first.set()
        holder.join(timeout=1)
    assert not holder.is_alive()


def test_owner_gate_admits_waiters_in_arrival_order() -> None:
    entered: list[int] = []
    first_started = threading.Event()
    release_first = threading.Event()

    def kernel(request: _Request) -> _Result:
        entered.append(request.value)
        if request.value == 1:
            first_started.set()
            assert release_first.wait(timeout=2)
        return _Result(value=request.value)

    tool = MathTool(
        operation_id="test.execution_admission.fifo",
        title="FIFO owner gate",
        description="Exercises FIFO execution-capacity admission.",
        request_type=_Request,
        result_type=_Result,
        run=kernel,
    )
    state = AppState(operation_catalog=Catalog((tool,)))
    outcomes: dict[int, OperationResult | Exception] = {}
    workers = [
        threading.Thread(
            target=_call,
            args=(state, tool.operation_id, value, threading.Event(), outcomes),
        )
        for value in (1, 2, 3)
    ]
    workers[0].start()
    assert first_started.wait(timeout=1)
    workers[1].start()
    time.sleep(0.05)
    workers[2].start()
    time.sleep(0.05)
    release_first.set()
    for worker_thread in workers:
        worker_thread.join(timeout=1)
        assert not worker_thread.is_alive()

    assert entered == [1, 2, 3]
    assert all(isinstance(outcome, OperationResult) for outcome in outcomes.values())
    second = outcomes[2]
    assert isinstance(second, OperationResult)
    assert second.queue_wait_ms >= 50
