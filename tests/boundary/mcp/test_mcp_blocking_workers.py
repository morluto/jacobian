"""MCP blocking-worker cancellation, drain, lease, and lifespan shutdown."""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from jacobian.adapters.mcp.context import (
    AppState,
    RuntimeLease,
    _runtime,
    _runtime_scope,
)
from jacobian.adapters.mcp.server import _runtime_lifespan
from jacobian.adapters.mcp.tooling import (
    MCPBlockingWorkerRegistry,
    MCPBlockingWorkerShutdownError,
    _run_blocking,
    blocking_worker_scope,
)


def test_cancelled_mcp_blocking_work_drains_before_request_task_finishes() -> None:
    release = threading.Event()
    finished = threading.Event()

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        cancellation_signalled = asyncio.Event()

        def blocking_operation() -> str:
            loop.call_soon_threadsafe(started.set)
            if not release.wait(timeout=2):
                raise AssertionError("test did not release blocking MCP work")
            finished.set()
            return "finished"

        task = asyncio.create_task(
            _run_blocking(
                blocking_operation,
                on_cancel=cancellation_signalled.set,
            )
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        await asyncio.wait_for(cancellation_signalled.wait(), timeout=1)
        assert cancellation_signalled.is_set()
        assert not task.done()
        assert not finished.is_set()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()

    asyncio.run(scenario())


def test_worker_draining_within_grace_is_awaited_and_persisted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker that completes within the grace period yields its drained result."""

    monkeypatch.setattr(
        "jacobian.adapters.mcp.tooling._CANCEL_DRAIN_GRACE_SECONDS", 1.0
    )
    release = threading.Event()
    finished = threading.Event()

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        cancellation_signalled = asyncio.Event()

        def blocking_operation() -> str:
            loop.call_soon_threadsafe(started.set)
            release.wait(timeout=2)
            finished.set()
            return "finished"

        task = asyncio.create_task(
            _run_blocking(
                blocking_operation,
                on_cancel=cancellation_signalled.set,
            )
        )
        try:
            await asyncio.wait_for(started.wait(), timeout=1)
            task.cancel()
            await asyncio.wait_for(cancellation_signalled.wait(), timeout=1)
            release.set()
            with pytest.raises(asyncio.CancelledError) as exc_info:
                await task
            assert finished.is_set()
            assert exc_info.value.drained_result == "finished"
        finally:
            release.set()

    asyncio.run(scenario())


def test_worker_that_does_not_drain_lets_cancellation_propagate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker that never completes is abandoned after the grace period."""

    monkeypatch.setattr(
        "jacobian.adapters.mcp.tooling._CANCEL_DRAIN_GRACE_SECONDS", 0.1
    )
    release = threading.Event()

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        cancellation_signalled = asyncio.Event()

        def blocking_operation() -> str:
            loop.call_soon_threadsafe(started.set)
            release.wait(timeout=5)
            return "finished"

        task = asyncio.create_task(
            _run_blocking(
                blocking_operation,
                on_cancel=cancellation_signalled.set,
            )
        )
        try:
            await asyncio.wait_for(started.wait(), timeout=1)
            task.cancel()
            await asyncio.wait_for(cancellation_signalled.wait(), timeout=1)
            cancel_time = time.monotonic()
            with pytest.raises(asyncio.CancelledError) as exc_info:
                await task
            assert exc_info.value.drained_result is None
            elapsed = time.monotonic() - cancel_time
            assert elapsed < 0.5, elapsed
        finally:
            release.set()

    asyncio.run(scenario())


def test_repeated_cancellation_does_not_extend_drain_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated cancellation during the drain does not reset the deadline."""

    monkeypatch.setattr(
        "jacobian.adapters.mcp.tooling._CANCEL_DRAIN_GRACE_SECONDS", 0.2
    )
    release = threading.Event()

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        first_cancellation = asyncio.Event()
        second_cancellation_dispatched = asyncio.Event()

        def blocking_operation() -> str:
            loop.call_soon_threadsafe(started.set)
            release.wait(timeout=5)
            return "finished"

        task = asyncio.create_task(
            _run_blocking(blocking_operation, on_cancel=first_cancellation.set)
        )
        try:
            await asyncio.wait_for(started.wait(), timeout=1)
            task.cancel()
            await asyncio.wait_for(first_cancellation.wait(), timeout=1)
            cancel_time = time.monotonic()
            await asyncio.sleep(0.1)
            task.cancel()
            loop.call_soon(second_cancellation_dispatched.set)
            await asyncio.wait_for(second_cancellation_dispatched.wait(), timeout=1)
            with pytest.raises(asyncio.CancelledError) as exc_info:
                await task
            assert exc_info.value.drained_result is None
            elapsed = time.monotonic() - cancel_time
            assert elapsed < 0.28, elapsed
        finally:
            release.set()

    asyncio.run(scenario())


def test_repeated_cancellation_keeps_draining_blocking_work() -> None:
    release = threading.Event()
    finished = threading.Event()

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        first_cancellation = asyncio.Event()
        second_cancellation_dispatched = asyncio.Event()

        def blocking_operation() -> str:
            loop.call_soon_threadsafe(started.set)
            if not release.wait(timeout=2):
                raise AssertionError("test did not release blocking MCP work")
            finished.set()
            return "finished"

        task = asyncio.create_task(
            _run_blocking(blocking_operation, on_cancel=first_cancellation.set)
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        await asyncio.wait_for(first_cancellation.wait(), timeout=1)
        task.cancel()
        loop.call_soon(second_cancellation_dispatched.set)
        await asyncio.wait_for(second_cancellation_dispatched.wait(), timeout=1)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()

    asyncio.run(scenario())


def test_cancelled_worker_retains_its_request_lease_until_late_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A late worker cannot outlive the tenant lease protecting its runtime."""

    monkeypatch.setattr(
        "jacobian.adapters.mcp.tooling._CANCEL_DRAIN_GRACE_SECONDS", 0.05
    )
    registry = MCPBlockingWorkerRegistry()
    release = threading.Event()
    lease_released = threading.Event()
    worker_finished = threading.Event()

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        started = asyncio.Event()

        def blocking_operation() -> None:
            loop.call_soon_threadsafe(started.set)
            release.wait(timeout=2)
            worker_finished.set()

        async def request() -> None:
            with blocking_worker_scope(registry, lease_release=lease_released.set):
                await _run_blocking(blocking_operation)

        task = asyncio.create_task(request())
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert registry.active_count == 1
        assert not lease_released.is_set()

        release.set()
        await asyncio.wait_for(asyncio.to_thread(worker_finished.wait, 1), timeout=1)
        await asyncio.sleep(0)
        assert lease_released.is_set()
        assert registry.active_count == 0

    asyncio.run(scenario())


def test_worker_registry_bounds_shutdown_and_consumes_late_results() -> None:
    registry = MCPBlockingWorkerRegistry()
    release = threading.Event()
    finished = threading.Event()

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        started = asyncio.Event()

        def blocking_operation() -> str:
            loop.call_soon_threadsafe(started.set)
            release.wait(timeout=2)
            finished.set()
            return "late result"

        async def request() -> None:
            with blocking_worker_scope(registry):
                await _run_blocking(blocking_operation)

        task = asyncio.create_task(request())
        await asyncio.wait_for(started.wait(), timeout=1)
        with pytest.raises(MCPBlockingWorkerShutdownError):
            await registry.close(timeout_seconds=0)
        assert registry.active_count == 1
        release.set()
        await task
        assert finished.is_set()
        await asyncio.sleep(0)
        assert registry.active_count == 0

    asyncio.run(scenario())


def test_timed_out_lifespan_shutdown_closes_owners_after_late_worker_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bounded shutdown retains, then deterministically releases its owners."""

    closed: list[str] = []

    class Runtime:
        def close(self) -> None:
            closed.append("runtime")

    class Router:
        def close(self) -> None:
            closed.append("router")

    def close_owners() -> None:
        Runtime().close()
        Router().close()

    class ImmediateShutdownRegistry(MCPBlockingWorkerRegistry):
        async def close(self) -> None:
            await super().close(timeout_seconds=0)

    monkeypatch.setattr(
        "jacobian.adapters.mcp.server._start_lean_warmup", lambda _: None
    )
    registry = ImmediateShutdownRegistry()

    async def scenario() -> None:
        release = asyncio.Event()

        async def late_worker() -> None:
            await release.wait()

        worker = asyncio.create_task(late_worker())
        registry.register(worker, request_scope=None)
        with pytest.raises(MCPBlockingWorkerShutdownError):
            state = AppState(
                acquire_runtime=lambda: RuntimeLease(Runtime()),  # type: ignore[arg-type]
                worker_registry=registry,
            )
            async with _runtime_lifespan(
                None,
                state=state,
                close_owner=close_owners,
            ):
                pass
        assert closed == []
        release.set()
        await worker
        await asyncio.sleep(0)
        assert closed == ["runtime", "router"]

    asyncio.run(scenario())


def test_failed_tenant_warmup_releases_the_acquired_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Warmup executes inside the scope which owns every tenant lease."""

    released = threading.Event()
    acquired = 0

    runtime = object()

    def acquire_runtime() -> RuntimeLease:
        nonlocal acquired
        acquired += 1
        return RuntimeLease(runtime, released.set)  # type: ignore[arg-type]

    state = AppState(
        acquire_runtime=acquire_runtime,
        worker_registry=MCPBlockingWorkerRegistry(),
    )
    ctx = SimpleNamespace(
        request_context=SimpleNamespace(lifespan_context=state),
    )
    monkeypatch.setattr(
        "jacobian.adapters.mcp.context._start_lean_warmup",
        lambda _: (_ for _ in ()).throw(RuntimeError("warmup failed")),
    )

    with (
        pytest.raises(RuntimeError, match="warmup failed"),
        _runtime(
            ctx  # type: ignore[arg-type]
        ),
    ):
        raise AssertionError("warmup failure must prevent request execution")

    assert acquired == 1
    assert released.is_set()


def test_injected_context_reuses_the_interceptor_tenant_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acquired = 0
    released = 0
    runtime = object()

    def release_runtime() -> None:
        nonlocal released
        released += 1

    def acquire_runtime() -> RuntimeLease:
        nonlocal acquired
        acquired += 1
        return RuntimeLease(runtime, release_runtime)  # type: ignore[arg-type]

    state = AppState(
        acquire_runtime=acquire_runtime,
        worker_registry=MCPBlockingWorkerRegistry(),
    )
    ctx = SimpleNamespace(
        request_context=SimpleNamespace(lifespan_context=state),
    )
    monkeypatch.setattr(
        "jacobian.adapters.mcp.context._start_lean_warmup",
        lambda _: None,
    )

    with (
        _runtime_scope(state) as interceptor_runtime,
        _runtime(
            ctx  # type: ignore[arg-type]
        ) as handler_runtime,
    ):
        assert interceptor_runtime is runtime
        assert handler_runtime is runtime

    assert acquired == 1
    assert released == 1
