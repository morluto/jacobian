"""Process-wide ownership of python-flint's mutable global context."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from threading import RLock
from time import monotonic

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    current_request_execution,
    request_cancelled,
)

_CONTEXT_LOCK_POLL_SECONDS = 0.1
_CONTEXT_LOCK = RLock()


@contextmanager
def flint_workprec(
    precision_bits: int, *, deadline: float | None = None
) -> Iterator[None]:
    """Own python-flint's global precision context for one bounded scope.

    ``deadline`` is an absolute ``time.monotonic`` instant. The earliest caller
    or request deadline bounds lock acquisition. A bound request cancellation
    interrupts a queued acquisition; callers without a deadline wait until the
    context is available while continuing to poll cancellation.
    """

    execution = current_request_execution()
    request_deadline = execution.deadline if execution is not None else None
    if deadline is not None and request_deadline is not None:
        wait_deadline = min(deadline, request_deadline)
    elif deadline is not None:
        wait_deadline = deadline
    elif request_deadline is not None:
        wait_deadline = request_deadline
    else:
        wait_deadline = None

    while True:
        if request_cancelled():
            raise OperationExecutionCancelledError(
                "operation cancelled waiting for the python-flint precision context"
            )
        remaining = wait_deadline - monotonic() if wait_deadline is not None else None
        if remaining is not None and remaining <= 0.0:
            raise OperationExecutionTimeoutError(
                "execution deadline expired waiting for the python-flint precision context"
            )
        poll_seconds = (
            min(_CONTEXT_LOCK_POLL_SECONDS, remaining)
            if remaining is not None
            else _CONTEXT_LOCK_POLL_SECONDS
        )
        if _CONTEXT_LOCK.acquire(timeout=poll_seconds):
            break
    try:
        if request_cancelled():
            raise OperationExecutionCancelledError(
                "operation cancelled waiting for the python-flint precision context"
            )
        from flint import ctx

        if wait_deadline is not None and monotonic() >= wait_deadline:
            raise OperationExecutionTimeoutError(
                "execution deadline expired waiting for the python-flint precision context"
            )
        with ctx.workprec(precision_bits):
            yield
    finally:
        _CONTEXT_LOCK.release()


__all__ = ["flint_workprec"]
