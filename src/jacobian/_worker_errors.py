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

_TIMEOUT_CONTEXT_FIELDS = {"configured_seconds", "adjustable_field_path"}
_MAX_TIMEOUT_PATH_COMPONENTS = 32
_MAX_TIMEOUT_PATH_COMPONENT_LENGTH = 128


def _decode_timeout_context(
    response: dict[str, object],
) -> tuple[float | None, tuple[str | int, ...] | None]:
    configured: float | None = None
    if "configured_seconds" in response:
        value = response["configured_seconds"]
        if not (type(value) is int or type(value) is float) or value < 0:
            raise ValueError("invalid configured timeout")
        try:
            finite = math.isfinite(value)
        except OverflowError as exc:
            raise ValueError("invalid configured timeout") from exc
        if not finite:
            raise ValueError("invalid configured timeout")
        configured = value
    path: tuple[str | int, ...] | None = None
    if "adjustable_field_path" in response:
        components = response["adjustable_field_path"]
        if (
            type(components) is not list
            or not components
            or len(components) > _MAX_TIMEOUT_PATH_COMPONENTS
        ):
            raise ValueError("invalid timeout field path")
        for component in components:
            if type(component) is str:
                if not 1 <= len(component) <= _MAX_TIMEOUT_PATH_COMPONENT_LENGTH:
                    raise ValueError("invalid timeout field path")
            elif type(component) is not int or not 0 <= component < 2**53:
                raise ValueError("invalid timeout field path")
        path = tuple(components)
    return configured, path


@contextmanager
def worker_execution_errors(
    *, preserve_timeout_context: bool = False
) -> Iterator[None]:
    """Encode classified failures; retain bounded backend causes privately.

    An owner may preserve source-authored timeout context after proving its
    complete error frame fits its existing stdout allowance. Other workers
    retain their compact error branch and unchanged channel requirements.
    """
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
        error: dict[str, object] = {"kind": "execution_error", "stage": exc.stage.value}
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
            if (
                isinstance(exc, OperationExecutionTimeoutError)
                and preserve_timeout_context
            ):
                if exc.configured_seconds is not None:
                    error["configured_seconds"] = exc.configured_seconds
                if exc.adjustable_field_path is not None:
                    path = exc.adjustable_field_path
                    if (
                        type(path) is not tuple
                        or not path
                        or len(path) > _MAX_TIMEOUT_PATH_COMPONENTS
                    ):
                        raise ValueError("invalid timeout field path") from None
                    error["adjustable_field_path"] = list(path)
                _decode_timeout_context(error)
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
        if response.get("reason") == "timeout" and set(response) <= (
            {"kind", "stage", "reason"} | _TIMEOUT_CONTEXT_FIELDS
        ):
            configured, path = _decode_timeout_context(response)
            raise OperationExecutionTimeoutError(
                "operation worker deadline expired",
                stage=stage,
                configured_seconds=configured,
                adjustable_field_path=path,
            )
        if (
            set(response) == {"kind", "stage", "reason"}
            and response["reason"] == "cancelled"
        ):
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
