"""Request-scoped state for local and remote MCP hosts."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from threading import Condition, Lock
from typing import Any

from mcp.server.mcpserver import Context

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationExecutionAdmission

_EXECUTION_QUEUE_POLL_SECONDS = 0.05
_DEFAULT_EXECUTION_QUEUE_WAIT_SECONDS = 30.0


class ExecutionAdmissionStatus(Enum):
    """One terminal outcome from bounded execution-capacity admission."""

    ADMITTED = "admitted"
    CANCELLED = "cancelled"
    BUSY = "busy"


@dataclass(frozen=True, slots=True)
class ExecutionPermit:
    """One acquired owner-local execution slot, or a terminal rejection."""

    status: ExecutionAdmissionStatus
    serialization_key: str | None
    waited_ms: int


class _ExecutionGate:
    """One FIFO, single-worker gate for a shared backend safety boundary."""

    def __init__(self) -> None:
        self._condition = Condition()
        self._queue: list[object] = []
        self._running = False

    def acquire(
        self,
        cancellation: Any,
        *,
        timeout_seconds: float,
        serialization_key: str,
    ) -> ExecutionPermit:
        started = time.monotonic()
        deadline = started + timeout_seconds
        token = object()
        with self._condition:
            self._queue.append(token)
            while True:
                if cancellation.is_set():
                    self._queue.remove(token)
                    self._condition.notify_all()
                    return ExecutionPermit(
                        status=ExecutionAdmissionStatus.CANCELLED,
                        serialization_key=None,
                        waited_ms=_elapsed_ms(started),
                    )
                if not self._running and self._queue[0] is token:
                    self._queue.pop(0)
                    self._running = True
                    return ExecutionPermit(
                        status=ExecutionAdmissionStatus.ADMITTED,
                        serialization_key=serialization_key,
                        waited_ms=_elapsed_ms(started),
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._queue.remove(token)
                    self._condition.notify_all()
                    return ExecutionPermit(
                        status=ExecutionAdmissionStatus.BUSY,
                        serialization_key=None,
                        waited_ms=_elapsed_ms(started),
                    )
                self._condition.wait(min(_EXECUTION_QUEUE_POLL_SECONDS, remaining))

    def release(self) -> None:
        with self._condition:
            if not self._running:
                raise RuntimeError("execution gate released without an admitted kernel")
            self._running = False
            self._condition.notify_all()


class ExecutionAdmissionController:
    """Admit requests according to immutable owner-declared safety scopes."""

    def __init__(self) -> None:
        self._gates_lock = Lock()
        self._gates: dict[str, _ExecutionGate] = {}

    def acquire(
        self,
        admission: OperationExecutionAdmission,
        cancellation: Any,
        *,
        timeout_seconds: float,
    ) -> ExecutionPermit:
        serialization_key = admission.serialization_key
        if serialization_key is None:
            status = (
                ExecutionAdmissionStatus.CANCELLED
                if cancellation.is_set()
                else ExecutionAdmissionStatus.ADMITTED
            )
            return ExecutionPermit(
                status=status,
                serialization_key=None,
                waited_ms=0,
            )
        with self._gates_lock:
            gate = self._gates.setdefault(serialization_key, _ExecutionGate())
        return gate.acquire(
            cancellation,
            timeout_seconds=timeout_seconds,
            serialization_key=serialization_key,
        )

    def release(self, permit: ExecutionPermit) -> None:
        serialization_key = permit.serialization_key
        if serialization_key is None:
            return
        with self._gates_lock:
            gate = self._gates.get(serialization_key)
        if gate is None:
            raise RuntimeError("execution permit refers to an unknown safety boundary")
        gate.release()


def _elapsed_ms(started: float) -> int:
    return max(0, round((time.monotonic() - started) * 1_000))


@dataclass(frozen=True, slots=True)
class AppState:
    operation_catalog: Catalog
    authorize: Callable[[], None] | None = None
    execution_admission: ExecutionAdmissionController = field(
        default_factory=ExecutionAdmissionController,
        repr=False,
        compare=False,
    )
    execution_queue_wait_seconds: float = _DEFAULT_EXECUTION_QUEUE_WAIT_SECONDS

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.execution_queue_wait_seconds)
            or self.execution_queue_wait_seconds <= 0
        ):
            raise ValueError("execution queue wait must be finite and positive")


class AuthenticationError(PermissionError):
    """A remote request lacks a usable authenticated tenant subject."""


def _state(ctx: Context[Any, Any]) -> AppState:
    """Return the AppState for the current request."""

    state = ctx.request_context.lifespan_context
    if not isinstance(state, AppState):
        raise RuntimeError(
            "Jacobian is unavailable for this request. Retry once; if it fails "
            "again, inspect the local Jacobian log."
        )
    return state


def _catalog(ctx: Context[Any, Any]) -> Catalog:
    """Return the serving catalog for discovery and inspection."""

    return _state(ctx).operation_catalog


def _authorize(ctx: Context[Any, Any]) -> None:
    """Run request-scoped authorization if the host provided one."""

    callback = _state(ctx).authorize
    if callback is not None:
        callback()


__all__ = ["AppState", "AuthenticationError"]
