"""Request-scoped execution envelope context."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class RequestCancellationSignal(Protocol):
    """Minimal cooperative cancellation signal bound to one request."""

    def is_set(self) -> bool: ...


class ProgressSink(Protocol):
    """Transport-neutral receiver for truthful monotone operation progress."""

    def report(
        self, progress: int, *, total: int | None = None, message: str | None = None
    ) -> None: ...


class TimeoutOwner(StrEnum):
    CALLER_DEADLINE = "caller_deadline"
    OPERATION_WALL = "operation_wall"
    BACKEND_TIMEOUT = "backend_timeout"


@dataclass(frozen=True, slots=True)
class OperationPhaseLease:
    """One admitted backend window with delivery and teardown time reserved."""

    operation_deadline: float
    backend_deadline: float
    delivery_reserve_seconds: float
    teardown_reserve_seconds: float


def lease_operation_phases(
    wall_seconds: float,
    *,
    admitted_response_bytes: int,
    validation_work: int,
) -> OperationPhaseLease:
    """Reserve calibrated mandatory post-backend phases before assigning its deadline."""

    if wall_seconds <= 0 or admitted_response_bytes < 0 or validation_work < 0:
        raise ValueError(
            "phase-lease inputs must be nonnegative with positive wall time"
        )
    execution = current_request_execution()
    started = execution.started_at if execution is not None else time.monotonic()
    operation_deadline = started + wall_seconds
    if execution is not None and execution.outer_deadline is not None:
        operation_deadline = min(operation_deadline, execution.outer_deadline)
    delivery_reserve = 0.02 + admitted_response_bytes / 20_000_000
    teardown_reserve = 0.02 + validation_work / 10_000_000
    backend_deadline = operation_deadline - delivery_reserve - teardown_reserve
    bind_request_deadline(operation_deadline)
    if backend_deadline <= time.monotonic():
        raise OperationExecutionTimeoutError(
            "operation has no remaining admitted backend lease",
            timeout_owner=TimeoutOwner.OPERATION_WALL,
            configured_seconds=wall_seconds,
            elapsed_seconds=max(0.0, time.monotonic() - started),
        )
    return OperationPhaseLease(
        operation_deadline=operation_deadline,
        backend_deadline=backend_deadline,
        delivery_reserve_seconds=delivery_reserve,
        teardown_reserve_seconds=teardown_reserve,
    )


@dataclass(frozen=True, slots=True)
class RequestExecutionEnvelope:
    """Immutable transport-neutral context for one complete request."""

    started_at: float
    outer_deadline: float | None = None
    timeout_owner: TimeoutOwner = TimeoutOwner.CALLER_DEADLINE
    cancellation_signal: RequestCancellationSignal | None = None
    progress_sink: ProgressSink | None = None

    @property
    def deadline(self) -> float | None:
        """Return the effective deadline for legacy owner reads."""

        operation_deadline = _OPERATION_DEADLINE.get()
        if self.outer_deadline is None:
            return operation_deadline
        if operation_deadline is None:
            return self.outer_deadline
        return min(self.outer_deadline, operation_deadline)


_REQUEST_EXECUTION: ContextVar[RequestExecutionEnvelope | None] = ContextVar(
    "jacobian_request_execution", default=None
)
_OPERATION_DEADLINE: ContextVar[float | None] = ContextVar(
    "jacobian_operation_deadline", default=None
)
_REQUEST_CANCELLATION: ContextVar[RequestCancellationSignal | None] = ContextVar(
    "jacobian_request_cancellation", default=None
)


@contextmanager
def request_execution(
    started_at: float,
    *,
    outer_deadline: float | None = None,
    timeout_owner: TimeoutOwner = TimeoutOwner.CALLER_DEADLINE,
    cancellation_signal: RequestCancellationSignal | None = None,
    progress_sink: ProgressSink | None = None,
) -> Iterator[RequestExecutionEnvelope]:
    """Bind one execution envelope across parsing, admission, and projection."""

    context = RequestExecutionEnvelope(
        started_at=started_at,
        outer_deadline=outer_deadline,
        timeout_owner=timeout_owner,
        cancellation_signal=cancellation_signal,
        progress_sink=progress_sink,
    )
    token: Token[RequestExecutionEnvelope | None] = _REQUEST_EXECUTION.set(context)
    deadline_token = _OPERATION_DEADLINE.set(None)
    try:
        yield context
    finally:
        _OPERATION_DEADLINE.reset(deadline_token)
        _REQUEST_EXECUTION.reset(token)


def current_request_execution() -> RequestExecutionEnvelope | None:
    """Return the current request envelope, if dispatch established one."""

    return _REQUEST_EXECUTION.get()


@contextmanager
def request_cancellation(event: RequestCancellationSignal) -> Iterator[None]:
    """Bind cooperative cancellation to the current request context."""

    token = _REQUEST_CANCELLATION.set(event)
    try:
        yield
    finally:
        _REQUEST_CANCELLATION.reset(token)


def current_request_cancellation() -> RequestCancellationSignal | None:
    """Return the current request cancellation signal, if one is bound."""

    return _REQUEST_CANCELLATION.get()


def request_cancelled() -> bool:
    """Report whether the current request has been cancelled."""

    envelope = current_request_execution()
    event = current_request_cancellation() or (
        envelope.cancellation_signal if envelope is not None else None
    )
    return event is not None and event.is_set()


def report_request_progress(
    progress: int, *, total: int | None = None, message: str | None = None
) -> None:
    """Report absolute truthful progress when the active transport requested it."""

    execution = current_request_execution()
    if execution is not None and execution.progress_sink is not None:
        execution.progress_sink.report(progress, total=total, message=message)


class OperationExecutionStage(StrEnum):
    """Bounded public phase for timeout and cancellation recovery."""

    REQUEST_PARSING = "request_parsing"
    OPERATION_EXECUTION = "operation_execution"
    RESULT_PROJECTION = "result_projection"


def request_checkpoint(
    stage: str, *, public_stage: OperationExecutionStage | None = None
) -> None:
    """Reject a cancelled or expired request at one documented execution stage."""

    if request_cancelled():
        raise OperationExecutionCancelledError(
            f"request cancelled {stage}", stage=public_stage or _public_stage(stage)
        )
    execution = current_request_execution()
    deadline = execution.deadline if execution is not None else None
    if execution is not None and deadline is not None:
        now = time.monotonic()
        if now < deadline:
            return
        raise OperationExecutionTimeoutError(
            f"request deadline expired {stage}",
            stage=public_stage or _public_stage(stage),
            timeout_owner=(
                execution.timeout_owner
                if execution.outer_deadline is not None
                and now >= execution.outer_deadline
                else TimeoutOwner.OPERATION_WALL
            ),
            elapsed_seconds=max(0.0, now - execution.started_at),
        )


def _public_stage(stage: str) -> OperationExecutionStage:
    if stage == "before parsing":
        return OperationExecutionStage.REQUEST_PARSING
    if "projection" in stage or "result construction" in stage:
        return OperationExecutionStage.RESULT_PROJECTION
    return OperationExecutionStage.OPERATION_EXECUTION


class OperationExecutionTimeoutError(TimeoutError):
    """The request-scoped owner envelope expired."""

    def __init__(
        self,
        message: str,
        *,
        stage: OperationExecutionStage = OperationExecutionStage.OPERATION_EXECUTION,
        timeout_owner: TimeoutOwner = TimeoutOwner.OPERATION_WALL,
        configured_seconds: float | None = None,
        elapsed_seconds: float | None = None,
        adjustable_field_path: tuple[str | int, ...] | None = None,
        maximum_seconds: float | None = None,
        deterministic_work_remains_fixed: bool = False,
    ) -> None:
        self.stage = stage
        self.timeout_owner = timeout_owner
        self.configured_seconds = configured_seconds
        self.elapsed_seconds = elapsed_seconds
        self.adjustable_field_path = adjustable_field_path
        self.maximum_seconds = maximum_seconds
        self.deterministic_work_remains_fixed = deterministic_work_remains_fixed
        super().__init__(message)


class OperationExecutionCancelledError(Exception):
    """The caller cancelled the request during a killable backend."""

    def __init__(
        self,
        message: str,
        *,
        stage: OperationExecutionStage = OperationExecutionStage.OPERATION_EXECUTION,
    ) -> None:
        self.stage = stage
        super().__init__(message)


class ExecutionResource(StrEnum):
    WORK = "work"
    MEMORY = "memory"
    OUTPUT = "output"


class BackendFailureReason(StrEnum):
    INITIALIZATION = "initialization"
    STARTUP = "startup"
    ABNORMAL_EXIT = "abnormal_exit"
    MALFORMED_RESPONSE = "malformed_response"
    INVALID_OUTPUT = "invalid_output"


class OperationResourceExhaustedError(Exception):
    """An admitted execution exhausted an operational capacity."""

    def __init__(
        self,
        resource: ExecutionResource,
        *,
        stage: OperationExecutionStage = OperationExecutionStage.OPERATION_EXECUTION,
    ) -> None:
        self.resource = resource
        self.stage = stage
        super().__init__(f"operation exhausted its {resource.value} allowance")


@dataclass(slots=True)
class OperationWorkLedger:
    """Charge deterministic kernel work before each bounded unit executes."""

    limit: int
    consumed: int = 0

    def charge(self, units: int = 1) -> None:
        if units < 0:
            raise ValueError("work charge must be nonnegative")
        if units > self.limit - self.consumed:
            raise OperationResourceExhaustedError(ExecutionResource.WORK)
        self.consumed += units


class OperationBackendError(Exception):
    """A backend failed to establish a usable mathematical result."""

    def __init__(
        self,
        reason: BackendFailureReason,
        *,
        stage: OperationExecutionStage = OperationExecutionStage.OPERATION_EXECUTION,
    ) -> None:
        self.reason = reason
        self.stage = stage
        super().__init__(f"operation backend failed ({reason.value})")


def bind_request_deadline(deadline: float) -> None:
    """Bind the operation deadline once without mutating its request envelope.

    Nested producers retain local backend deadlines; they cannot shorten or
    later widen the enclosing operation's deadline.
    """

    if _REQUEST_EXECUTION.get() is not None and _OPERATION_DEADLINE.get() is None:
        _OPERATION_DEADLINE.set(deadline)


def execution_deadline(seconds: float) -> float:
    """Bind the owner's allowance once, including earlier dispatch phases."""
    execution = current_request_execution()
    start = execution.started_at if execution is not None else time.monotonic()
    deadline = start + seconds
    if execution is not None and execution.deadline is not None:
        deadline = min(deadline, execution.deadline)
    bind_request_deadline(deadline)
    require_execution_deadline(deadline)
    return deadline


def require_execution_deadline(deadline: float) -> None:
    request_checkpoint("during operation execution")
    if time.monotonic() >= deadline:
        execution = current_request_execution()
        raise OperationExecutionTimeoutError(
            "operation deadline expired",
            configured_seconds=(
                max(0.0, deadline - execution.started_at)
                if execution is not None
                else None
            ),
            elapsed_seconds=(
                max(0.0, time.monotonic() - execution.started_at)
                if execution is not None
                else None
            ),
        )


def remaining_timeout_ms(timeout_ms: int) -> int:
    """Give a mandatory solver phase only the unspent request allowance."""
    request_checkpoint("before solver phase")
    execution = current_request_execution()
    if execution is None or execution.deadline is None:
        return timeout_ms
    return min(timeout_ms, max(1, int((execution.deadline - time.monotonic()) * 1000)))


__all__ = [
    "BackendFailureReason",
    "ExecutionResource",
    "OperationBackendError",
    "OperationExecutionCancelledError",
    "OperationExecutionStage",
    "OperationExecutionTimeoutError",
    "OperationPhaseLease",
    "OperationResourceExhaustedError",
    "OperationWorkLedger",
    "ProgressSink",
    "RequestCancellationSignal",
    "RequestExecutionEnvelope",
    "TimeoutOwner",
    "bind_request_deadline",
    "current_request_cancellation",
    "current_request_execution",
    "execution_deadline",
    "lease_operation_phases",
    "remaining_timeout_ms",
    "report_request_progress",
    "request_cancellation",
    "request_cancelled",
    "request_checkpoint",
    "request_execution",
    "require_execution_deadline",
]
