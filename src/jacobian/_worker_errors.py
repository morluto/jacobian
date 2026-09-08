"""Private execution-error branch for bounded mathematical workers."""

from __future__ import annotations

import json
import math
import traceback
from collections.abc import Iterator
from contextlib import contextmanager

from jacobian._execution import (
    BackendFailureReason,
    ExecutionResource,
    OperationBackendError,
    OperationExecutionCancelledError,
    OperationExecutionStage,
    OperationExecutionTimeoutError,
    OperationResourceExhaustedError,
)


@contextmanager
def worker_execution_errors() -> Iterator[None]:
    """Encode classified failures; retain bounded backend causes privately."""
    try:
        try:
            yield
        except ImportError as exc:
            raise OperationBackendError(BackendFailureReason.INITIALIZATION) from exc
        except MemoryError as exc:
            raise OperationResourceExhaustedError(ExecutionResource.MEMORY) from exc
    except (
        OperationBackendError,
        OperationResourceExhaustedError,
        OperationExecutionTimeoutError,
        OperationExecutionCancelledError,
    ) as exc:
        error = {"kind": "execution_error", "stage": exc.stage.value}
        if isinstance(exc, OperationBackendError):
            error["reason"] = exc.reason.value
            error["diagnostic"] = "".join(traceback.format_exception(exc))[-1024:]
        elif isinstance(exc, OperationResourceExhaustedError):
            error["resource"] = exc.resource.value
        else:
            error["reason"] = (
                "timeout"
                if isinstance(exc, OperationExecutionTimeoutError)
                else "cancelled"
            )
        print(json.dumps(error, separators=(",", ":")))


def decode_worker_execution_error(response: object) -> None:
    """Raise a validated private execution branch, leaving mathematics to owners."""
    if not isinstance(response, dict) or response.get("kind") != "execution_error":
        return
    try:
        stage = OperationExecutionStage(response["stage"])
        if set(response) == {"kind", "stage", "resource"}:
            raise OperationResourceExhaustedError(
                ExecutionResource(response["resource"]), stage=stage
            )
        if set(response) == {"kind", "stage", "reason"}:
            if response["reason"] == "timeout":
                raise OperationExecutionTimeoutError(
                    "operation worker deadline expired", stage=stage
                )
            if response["reason"] == "cancelled":
                raise OperationExecutionCancelledError(
                    "operation worker cancelled", stage=stage
                )
        if set(response) == {"kind", "stage", "reason", "diagnostic"}:
            reason = BackendFailureReason(response["reason"])
            diagnostic = response["diagnostic"]
            if not isinstance(diagnostic, str) or len(diagnostic) > 1024:
                raise ValueError("invalid private diagnostic")
            error = OperationBackendError(reason, stage=stage)
            error.add_note(diagnostic)
            raise error
        raise ValueError("invalid execution-error branch")
    except (KeyError, TypeError, ValueError) as exc:
        raise OperationBackendError(BackendFailureReason.MALFORMED_RESPONSE) from exc


def bind_worker_deadline(payload: object) -> None:
    """Consume the private parent deadline before parsing mathematical input."""
    from jacobian._execution import bind_request_deadline, request_checkpoint

    if not isinstance(payload, dict):
        raise ValueError("worker payload must be an object")
    deadline = payload.pop("_deadline", None)
    if type(deadline) not in (float, int) or not math.isfinite(deadline):
        raise ValueError("worker payload requires a finite deadline")
    bind_request_deadline(deadline)
    request_checkpoint("before worker request parsing")
