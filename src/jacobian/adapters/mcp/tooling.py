"""MCP tool invocation helpers: annotations, logging, and capability attempts."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any

from mcp_types import ToolAnnotations

from jacobian.adapters.mcp.projections import (
    _invoke_capability_with_cancellation,
)
from jacobian.canonical import canonicalize_json
from jacobian.contracts.capabilities import (
    CapabilityRequest,
    CapabilityResult,
)

_LOGGER = logging.getLogger(__name__)

_CANCEL_DRAIN_GRACE_SECONDS = 5.0
_SHUTDOWN_DRAIN_GRACE_SECONDS = 5.0


class MCPBlockingWorkerShutdownError(RuntimeError):
    """MCP lifespan shutdown reached its worker-drain deadline."""


class MCPBlockingWorkerRegistry:
    """Lifespan-owned ownership registry for blocking MCP worker tasks.

    A cancelled request may finish before its thread does.  The registry keeps
    that task observable until its result or failure is consumed, and lets a
    tenant request lease remain held for the same interval.
    """

    def __init__(self) -> None:
        self._workers: set[asyncio.Task[Any]] = set()
        self._closing = False
        self._quiescence_callbacks: list[Callable[[], None]] = []

    @property
    def active_count(self) -> int:
        # A finished task still owns its request lease until its done callback
        # consumes the late result and releases that scope.
        return len(self._workers)

    def register(
        self,
        worker: asyncio.Task[Any],
        *,
        request_scope: _MCPBlockingRequestScope | None,
    ) -> None:
        """Retain a worker until its terminal result has been consumed."""

        if self._closing:
            raise MCPBlockingWorkerShutdownError("MCP server is shutting down")
        self._workers.add(worker)
        if request_scope is not None:
            request_scope.retain_worker()

        def consume_late_result(done: asyncio.Task[Any]) -> None:
            self._workers.discard(done)
            if request_scope is not None:
                request_scope.release_worker()
            try:
                done.result()
            except asyncio.CancelledError:
                _LOGGER.debug("blocking MCP worker was cancelled during cleanup")
            except BaseException:
                _LOGGER.debug(
                    "blocking MCP worker failed after its request ended",
                    exc_info=True,
                )
            finally:
                self._run_quiescence_callbacks()

        worker.add_done_callback(consume_late_result)

    def defer_until_quiescent(self, callback: Callable[[], None]) -> None:
        """Run cleanup after every retained worker has released its request scope.

        Lifespan shutdown uses this after a bounded drain expires.  The callback
        runs on the event-loop thread after the final worker's late result has
        been consumed, so tenant leases protecting its runtime have already
        been released.
        """

        if self.active_count == 0:
            self._run_cleanup_callback(callback)
            return
        self._quiescence_callbacks.append(callback)

    def _run_quiescence_callbacks(self) -> None:
        if self.active_count != 0:
            return
        callbacks, self._quiescence_callbacks = self._quiescence_callbacks, []
        for callback in callbacks:
            self._run_cleanup_callback(callback)

    @staticmethod
    def _run_cleanup_callback(callback: Callable[[], None]) -> None:
        try:
            callback()
        except BaseException:
            _LOGGER.exception("deferred MCP worker cleanup failed")

    async def close(
        self, *, timeout_seconds: float = _SHUTDOWN_DRAIN_GRACE_SECONDS
    ) -> None:
        """Bound lifespan shutdown while retaining late-worker cleanup ownership."""

        if timeout_seconds < 0:
            raise ValueError("timeout_seconds must be non-negative")
        self._closing = True
        workers = tuple(self._workers)
        if not workers:
            return
        _done, pending = await asyncio.wait(workers, timeout=timeout_seconds)
        if pending:
            raise MCPBlockingWorkerShutdownError(
                "blocking MCP workers did not quiesce before server shutdown"
            )
        # ``asyncio.wait`` observes task completion before every registered
        # callback has necessarily run.  Yield once so scopes are released
        # before a synchronous lifecycle close is allowed to proceed.
        await asyncio.sleep(0)


class _MCPBlockingRequestScope:
    """Release a request's tenant lease only after all of its workers finish."""

    def __init__(self, lease_release: Callable[[], None] | None) -> None:
        self._lease_release = lease_release
        self._workers = 0
        self._request_finished = False
        self._released = False

    def retain_worker(self) -> None:
        self._workers += 1

    def release_worker(self) -> None:
        self._workers -= 1
        self._release_if_ready()

    def finish_request(self) -> None:
        self._request_finished = True
        self._release_if_ready()

    def _release_if_ready(self) -> None:
        if (
            self._request_finished
            and self._workers == 0
            and not self._released
            and self._lease_release is not None
        ):
            self._released = True
            self._lease_release()


_blocking_worker_scope: ContextVar[_MCPBlockingRequestScope | None] = ContextVar(
    "jacobian_mcp_blocking_worker_scope",
    default=None,
)
_blocking_worker_registry: ContextVar[MCPBlockingWorkerRegistry | None] = ContextVar(
    "jacobian_mcp_blocking_worker_registry",
    default=None,
)


@contextmanager
def blocking_worker_scope(
    registry: MCPBlockingWorkerRegistry,
    *,
    lease_release: Callable[[], None] | None = None,
) -> Any:
    """Bind lifespan worker ownership to one MCP request or resource read."""

    scope = _MCPBlockingRequestScope(lease_release)
    registry_token: Token[MCPBlockingWorkerRegistry | None] = (
        _blocking_worker_registry.set(registry)
    )
    scope_token: Token[_MCPBlockingRequestScope | None] = _blocking_worker_scope.set(
        scope
    )
    try:
        yield
    finally:
        _blocking_worker_scope.reset(scope_token)
        _blocking_worker_registry.reset(registry_token)
        scope.finish_request()


async def _drain_cancelled_worker[BlockingResultT](
    worker: asyncio.Task[BlockingResultT],
) -> BlockingResultT | None:
    """Wait for a cancelled worker up to the drain grace period.

    Returns the worker result if it completes within the grace period,
    otherwise ``None`` after logging that the worker did not drain.
    """

    drain_deadline = time.monotonic() + _CANCEL_DRAIN_GRACE_SECONDS
    while not worker.done():
        remaining = drain_deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            return await asyncio.wait_for(asyncio.shield(worker), timeout=remaining)
        except TimeoutError:
            break
        except asyncio.CancelledError:
            continue
        except BaseException:
            _LOGGER.debug(
                "blocking MCP worker failed while its cancelled request drained",
                exc_info=True,
            )
            return None
    if not worker.done():
        _LOGGER.warning(
            "blocking MCP worker did not drain within %.1fs grace period",
            _CANCEL_DRAIN_GRACE_SECONDS,
        )
        return None
    try:
        return worker.result()
    except BaseException:
        _LOGGER.debug(
            "blocking MCP worker failed while its cancelled request drained",
            exc_info=True,
        )
        return None


async def _run_blocking[BlockingResultT](
    function: Callable[..., BlockingResultT],
    /,
    *args: Any,
    on_cancel: Callable[[], None] | None = None,
) -> BlockingResultT:
    """Run blocking MCP work without detaching it from request teardown.

    On cancellation, waits for the worker to drain up to
    ``_CANCEL_DRAIN_GRACE_SECONDS`` and stores the result in the
    ``drained_result`` attribute so the caller can persist a completed
    outcome instead of unconditionally recording cancellation.  If the
    worker does not finish within the grace period, the caller is
    unblocked without a drained result.
    """

    worker = asyncio.create_task(asyncio.to_thread(function, *args))
    registry = _blocking_worker_registry.get()
    if registry is not None:
        try:
            registry.register(worker, request_scope=_blocking_worker_scope.get())
        except BaseException:
            worker.cancel()
            raise
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError as exc:
        if on_cancel is not None:
            on_cancel()
        drained = await _drain_cancelled_worker(worker)
        exc.drained_result = drained  # type: ignore[attr-defined]
        raise


class AgentRecoveryError(RuntimeError):
    """A safe, actionable failure intended for an agent tool response."""


def _tool_annotations(
    *,
    read_only: bool = False,
    destructive: bool = False,
    idempotent: bool = False,
) -> ToolAnnotations:
    return ToolAnnotations(
        read_only_hint=read_only,
        destructive_hint=destructive,
        idempotent_hint=idempotent,
        open_world_hint=False,
    )


def _argument_digest(arguments: dict[str, Any]) -> str:
    try:
        encoded = canonicalize_json(arguments)
    except (TypeError, ValueError):
        encoded = json.dumps(
            arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _request_trace_digest(ctx: Any | None) -> tuple[str, str]:
    """Return a bounded correlation digest without retaining caller identifiers."""

    if ctx is None:
        return "none", "none"
    headers = getattr(ctx, "headers", None)
    if headers is not None:
        traceparent = headers.get("traceparent")
        if isinstance(traceparent, str) and 0 < len(traceparent) <= 256:
            digest = hashlib.sha256(traceparent.encode("utf-8")).hexdigest()[:8]
            return digest, "traceparent"
    try:
        request_id = str(ctx.request_id)
    except (AttributeError, TypeError, ValueError):
        return "none", "none"
    if not request_id or len(request_id) > 256:
        return "none", "none"
    digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:8]
    return digest, "request_id"


def _request_id_digest(ctx: Any | None) -> str:
    """Return a bounded per-request correlation digest for telemetry joins."""

    if ctx is None:
        return "none"
    try:
        request_id = str(ctx.request_id)
    except (AttributeError, TypeError, ValueError):
        return "none"
    if not request_id or len(request_id) > 256:
        return "none"
    return hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:16]


def _response_size(value: Any) -> int:
    try:
        if hasattr(value, "model_dump_json"):
            return len(value.model_dump_json().encode("utf-8"))
        return len(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        )
    except (TypeError, ValueError):
        return -1


def _log_capability_attempt(
    *,
    capability_id: str,
    started: float,
    argument_digest: str,
    request_digest: str,
    trace_digest: str,
    trace_source: str,
    result: CapabilityResult | None = None,
    execution_status: str | None = None,
    diagnostic_codes: tuple[str, ...] = (),
) -> None:
    if result is not None:
        execution_status = result.execution.status.value
        diagnostic_codes = tuple(item.code for item in result.diagnostics)
        capability_version = result.capability_version
        verification_record_uri_present = result.verification_record_uri is not None
        operation_runtime_ms = result.execution.runtime_ms
        response_bytes = _response_size(result)
    else:
        capability_version = "unknown"
        verification_record_uri_present = False
        operation_runtime_ms = None
        response_bytes = 0
    codes = ",".join(diagnostic_codes[:8]) or "none"
    _LOGGER.info(
        "MCP capability attempt request_digest=%s trace_digest=%s trace_source=%s "
        "capability_id=%s capability_version=%s "
        "execution_status=%s verification_record_uri_present=%s diagnostic_codes=%s "
        "attempt_duration_ms=%.3f operation_runtime_ms=%s "
        "response_bytes=%d argument_digest=%s",
        request_digest,
        trace_digest,
        trace_source,
        capability_id,
        capability_version,
        execution_status or "ERROR",
        verification_record_uri_present,
        codes,
        (time.monotonic() - started) * 1000,
        "none" if operation_runtime_ms is None else operation_runtime_ms,
        response_bytes,
        argument_digest,
    )


async def _invoke_capability_attempt(
    runtime: Any,
    *,
    capability_id: str,
    payload: dict[str, Any],
    inputs: dict[str, str] | None = None,
    ctx: Any | None,
) -> CapabilityResult:
    started = time.monotonic()
    argument_digest = _argument_digest(
        {
            "capability_id": capability_id,
            "payload": payload,
            "inputs": inputs or {},
        }
    )
    trace_digest, trace_source = _request_trace_digest(ctx)
    request_digest = _request_id_digest(ctx)
    cancellation_event = threading.Event()
    request = CapabilityRequest(
        capability_id=capability_id,
        input=payload,
        inputs=inputs or {},
    )
    try:
        result = await _run_blocking(
            _invoke_capability_with_cancellation,
            runtime,
            request,
            cancellation_event,
            on_cancel=cancellation_event.set,
        )
    except asyncio.CancelledError as exc:
        drained = getattr(exc, "drained_result", None)
        if isinstance(drained, CapabilityResult):
            _log_capability_attempt(
                capability_id=capability_id,
                started=started,
                argument_digest=argument_digest,
                request_digest=request_digest,
                trace_digest=trace_digest,
                trace_source=trace_source,
                result=drained,
            )
        else:
            _log_capability_attempt(
                capability_id=capability_id,
                started=started,
                argument_digest=argument_digest,
                request_digest=request_digest,
                trace_digest=trace_digest,
                trace_source=trace_source,
                execution_status="CANCELLED",
                diagnostic_codes=("CLIENT_CANCELLED",),
            )
        raise
    except Exception:
        _log_capability_attempt(
            capability_id=capability_id,
            started=started,
            argument_digest=argument_digest,
            request_digest=request_digest,
            trace_digest=trace_digest,
            trace_source=trace_source,
            execution_status="ERROR",
            diagnostic_codes=("INVOCATION_EXCEPTION",),
        )
        raise
    _log_capability_attempt(
        capability_id=capability_id,
        started=started,
        argument_digest=argument_digest,
        request_digest=request_digest,
        trace_digest=trace_digest,
        trace_source=trace_source,
        result=result,
    )
    return result
