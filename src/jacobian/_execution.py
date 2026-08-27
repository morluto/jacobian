"""Request-scoped execution envelope context."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class RequestExecution:
    """One request's start instant and owner-bound deadline."""

    started_at: float
    deadline: float | None = None


_REQUEST_EXECUTION: ContextVar[RequestExecution | None] = ContextVar(
    "jacobian_request_execution", default=None
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


def bind_request_deadline(deadline: float) -> None:
    """Attach an owner-derived deadline to the current request envelope."""

    context = _REQUEST_EXECUTION.get()
    if context is not None:
        _REQUEST_EXECUTION.set(replace(context, deadline=deadline))


__all__ = [
    "RequestExecution",
    "bind_request_deadline",
    "current_request_execution",
    "request_execution",
]
