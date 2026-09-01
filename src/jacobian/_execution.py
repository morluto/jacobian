"""Request-scoped execution envelope context."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, replace
from typing import Protocol


class RequestCancellationSignal(Protocol):
    """Minimal cooperative cancellation signal bound to one request."""

    def is_set(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class RequestExecution:
    """One request's start instant and owner-bound deadline."""

    started_at: float
    deadline: float | None = None


_REQUEST_EXECUTION: ContextVar[RequestExecution | None] = ContextVar(
    "jacobian_request_execution", default=None
)
_REQUEST_CANCELLATION: ContextVar[RequestCancellationSignal | None] = ContextVar(
    "jacobian_request_cancellation", default=None
)


@contextmanager
def request_execution(started_at: float) -> Iterator[RequestExecution]:
    """Bind one execution envelope across parsing, admission, and projection."""

    context = RequestExecution(started_at=started_at)
    token: Token[RequestExecution | None] = _REQUEST_EXECUTION.set(context)
    try:
        yield context
    finally:
        _REQUEST_EXECUTION.reset(token)


def current_request_execution() -> RequestExecution | None:
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

    event = current_request_cancellation()
    return event is not None and event.is_set()


def request_checkpoint(stage: str) -> None:
    """Reject a cancelled or expired request at one documented execution stage."""

    if request_cancelled():
        raise OperationExecutionCancelledError(f"request cancelled {stage}")
    execution = current_request_execution()
    if (
        execution is not None
        and execution.deadline is not None
        and time.monotonic() >= execution.deadline
    ):
        raise OperationExecutionTimeoutError(f"request deadline expired {stage}")


class OperationExecutionTimeoutError(TimeoutError):
    """The request-scoped owner envelope expired."""


class OperationExecutionCancelledError(Exception):
    """The caller cancelled the request during a killable backend."""


def bind_request_deadline(deadline: float) -> None:
    """Attach an owner-derived deadline to the current request envelope."""

    context = _REQUEST_EXECUTION.get()
    if context is not None:
        _REQUEST_EXECUTION.set(replace(context, deadline=deadline))


__all__ = [
    "OperationExecutionCancelledError",
    "OperationExecutionTimeoutError",
    "RequestCancellationSignal",
    "RequestExecution",
    "bind_request_deadline",
    "current_request_cancellation",
    "current_request_execution",
    "request_cancellation",
    "request_cancelled",
    "request_checkpoint",
    "request_execution",
]
