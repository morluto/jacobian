"""Process-wide ownership of python-flint's mutable global context."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from threading import RLock
from time import monotonic

from jacobian._execution import (
    OperationExecutionTimeoutError,
    current_request_execution,
)

_DEFAULT_CONTEXT_WAIT_SECONDS = 120.0
_CONTEXT_LOCK = RLock()


@contextmanager
def flint_workprec(
    precision_bits: int, *, deadline: float | None = None
) -> Iterator[None]:
    """Own python-flint's global precision context for one bounded scope.

    ``deadline`` is an absolute ``time.monotonic`` instant. The earliest caller
    or request deadline bounds lock acquisition. Callers without either get a
    120-second lock-wait ceiling, matching existing bounded backend adapters.
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
        wait_deadline = monotonic() + _DEFAULT_CONTEXT_WAIT_SECONDS

    remaining = wait_deadline - monotonic()
    if remaining <= 0.0 or not _CONTEXT_LOCK.acquire(timeout=remaining):
        raise OperationExecutionTimeoutError(
            "execution deadline expired waiting for the python-flint precision context"
        )
    try:
        from flint import ctx

        if monotonic() >= wait_deadline:
            raise OperationExecutionTimeoutError(
                "execution deadline expired waiting for the python-flint precision context"
            )
        with ctx.workprec(precision_bits):
            yield
    finally:
        _CONTEXT_LOCK.release()


__all__ = ["flint_workprec"]
