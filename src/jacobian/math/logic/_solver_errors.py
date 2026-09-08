"""Shared Z3 execution classification, excluding caller source diagnostics."""

import re
from typing import Literal, Never

from jacobian._execution import (
    ExecutionResource,
    OperationExecutionTimeoutError,
    OperationResourceExhaustedError,
)

_UnknownResource = Literal["time", "work", "memory"]
_Z3_SOURCE_DIAGNOSTIC = re.compile(r'\(error "line \d+ column \d+: ')


def _raise_exhaustion(
    resource: _UnknownResource, *, cause: Exception | None = None
) -> Never:
    if resource == "time":
        raise OperationExecutionTimeoutError(
            "operation solver time allowance expired"
        ) from cause
    raise OperationResourceExhaustedError(ExecutionResource(resource)) from cause


def _classify_exhaustion(message: str) -> _UnknownResource | None:
    """Classify one Z3 reason or exception message onto the exhausted budgets.

    Exhaustion keywords classify only backend conditions, which carry no
    source locator. A message containing a located ``(error "line ...
    column ...: ...")`` diagnostic is never classified as exhaustion, even
    when its text mentions a resource keyword: the diagnostic quotes
    caller-controlled source spellings, so an undeclared identifier named
    ``memory`` or a comment mentioning ``timeout`` must not report an
    exhausted budget.
    """

    if _Z3_SOURCE_DIAGNOSTIC.search(message) is not None:
        return None
    lowered = message.strip().lower()
    if "resource limit" in lowered or "canceled" in lowered:
        return "work"
    if "memory" in lowered:
        return "memory"
    if "timeout" in lowered or "time limit" in lowered:
        return "time"
    return None


def _project_unknown(reason: str | None) -> tuple[_UnknownResource | None, str]:
    """Project one Z3 unknown reason onto the typed exhausted-budget taxonomy."""

    text = (reason or "").strip()
    classified = _classify_exhaustion(text)
    if classified is not None:
        _raise_exhaustion(classified)
    if not text:
        return None, "the solver returned no completeness evidence"
    return None, "the solver returned an inconclusive answer"
