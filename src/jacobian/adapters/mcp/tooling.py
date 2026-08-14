"""MCP tool invocation helpers: annotations, logging, and operation attempts."""

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
    _invoke_operation_with_cancellation,
)
from jacobian.canonical import canonicalize_json
from jacobian.contracts.operations import (
    OperationRequest,
    OperationResult,
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
    tenant runtime hold remain active for the same interval.
    """

    def __init__(self) -> None:
        self._workers: set[asyncio.Task[Any]] = set()
        self._closing = False
        self._quiescence_callbacks: list[Callable[[], None]] = []

    @property
    def active_count(self) -> int:
        # A finished task still owns its request hold until its done callback
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
        been consumed, so tenant holds protecting its runtime have already
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
    """Release a tenant runtime only after all request workers finish."""

    def __init__(self, release_callback: Callable[[], None] | None) -> None:
        self._release_callback = release_callback
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
            and self._release_callback is not None
        ):
            self._released = True
            self._release_callback()


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
    release_callback: Callable[[], None] | None = None,
) -> Any:
    """Bind lifespan worker ownership to one MCP request or resource read."""

    scope = _MCPBlockingRequestScope(release_callback)
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


def _log_operation_attempt(
    *,
    operation_id: str,
    request_digest: str,
    provider: str,
    checker_ids: tuple[str, ...],
    result: OperationResult | None = None,
    execution_status: str | None = None,
    diagnostic_codes: tuple[str, ...] = (),
) -> None:
    if result is not None:
        execution_status = result.execution.status.value
        diagnostic_codes = tuple(item.code for item in result.diagnostics)
        operation_version = result.operation_version
        verification_record_uri_present = result.verification_record_uri is not None
        artifact_count = len(result.artifact_uris)
    else:
        operation_version = "unknown"
        verification_record_uri_present = False
        artifact_count = 0
    codes = ",".join(diagnostic_codes[:8]) or "none"
    checkers = ",".join(checker_ids) or "none"
    _LOGGER.info(
        "MCP operation attempt request_digest=%s operation_id=%s "
        "operation_version=%s provider=%s checker_ids=%s "
        "execution_status=%s verification_record_uri_present=%s diagnostic_codes=%s "
        "artifact_count=%d",
        request_digest,
        operation_id,
        operation_version,
        provider,
        checkers,
        execution_status or "ERROR",
        verification_record_uri_present,
        codes,
        artifact_count,
    )


async def _invoke_operation_attempt(
    runtime: Any,
    *,
    operation_id: str,
    payload: dict[str, Any],
    inputs: dict[str, str] | None = None,
) -> OperationResult:
    request_digest = _argument_digest(
        {
            "operation_id": operation_id,
            "payload": payload,
            "inputs": inputs or {},
        }
    )
    descriptor = runtime.core.operations.inspect(operation_id)
    provider = descriptor.provider if descriptor is not None else "unknown"
    checker_ids = (
        tuple(str(checker_id) for checker_id in descriptor.provider_runtime.checker_ids)
        if descriptor is not None and descriptor.provider_runtime is not None
        else ()
    )
    cancellation_event = threading.Event()
    request = OperationRequest(
        operation_id=operation_id,
        input=payload,
        inputs=inputs or {},
    )
    try:
        result = await _run_blocking(
            _invoke_operation_with_cancellation,
            runtime,
            request,
            cancellation_event,
            on_cancel=cancellation_event.set,
        )
    except asyncio.CancelledError as exc:
        drained = getattr(exc, "drained_result", None)
        if isinstance(drained, OperationResult):
            _log_operation_attempt(
                operation_id=operation_id,
                request_digest=request_digest,
                provider=provider,
                checker_ids=checker_ids,
                result=drained,
            )
        else:
            _log_operation_attempt(
                operation_id=operation_id,
                request_digest=request_digest,
                provider=provider,
                checker_ids=checker_ids,
                execution_status="CANCELLED",
                diagnostic_codes=("CLIENT_CANCELLED",),
            )
        raise
    except Exception:
        _log_operation_attempt(
            operation_id=operation_id,
            request_digest=request_digest,
            provider=provider,
            checker_ids=checker_ids,
            execution_status="ERROR",
            diagnostic_codes=("INVOCATION_EXCEPTION",),
        )
        raise
    _log_operation_attempt(
        operation_id=operation_id,
        request_digest=request_digest,
        provider=provider,
        checker_ids=checker_ids,
        result=result,
    )
    return result
