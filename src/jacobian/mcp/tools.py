"""MCP tool handlers for the operation surface."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp.shared.exceptions import MCPError
from mcp.types import INVALID_PARAMS

from jacobian._execution import request_execution
from jacobian.catalog.models import OperationId, OperationResult
from jacobian.dispatch import (
    OperationDomainValidationError,
    OperationExecutionTimeoutError,
    OperationRequestValidationError,
    _invoke_prepared_operation,
    _OperationResolutionError,
    _prepare_operation,
)
from jacobian.mcp.models import (
    OperationBrowseRequest,
    OperationFindRequest,
    OperationFindResponse,
    OperationInspectionResult,
    OperationInvalidRequestData,
    OperationSearchRequest,
    OperationValidationIssue,
)
from jacobian.mcp.projections import (
    _operation_browse_response,
    _operation_discovery_response,
)
from jacobian.mcp.runtime import (
    AppState,
    _authorize,
    _catalog,
)
from jacobian.process import bounded_process_cancellation

_MAX_VALIDATION_ERRORS = 64
_MAX_VALIDATION_LOCATION_COMPONENTS = 32
_MAX_VALIDATION_LOCATION_LENGTH = 128


class _CancellationSignal(Protocol):
    """Cooperative request cancellation observed by external process work."""

    def is_set(self) -> bool: ...


def math_find(
    request: OperationFindRequest,
    *,
    ctx: Context[AppState, Any],
) -> OperationFindResponse:
    active_catalog = _catalog(ctx)
    if isinstance(request, OperationSearchRequest):
        discovery_response = _operation_discovery_response(
            active_catalog,
            query=request.query,
            namespace=request.namespace,
            limit=request.limit,
            cursor=request.cursor,
        )
        return OperationFindResponse.model_validate(discovery_response)
    if isinstance(request, OperationBrowseRequest):
        browse_response = _operation_browse_response(
            active_catalog,
            namespace=request.namespace,
            limit=request.limit,
            cursor=request.cursor,
        )
        return OperationFindResponse.model_validate(browse_response)
    operation_id = request.operation_id
    descriptor = active_catalog.inspect(operation_id)
    if descriptor is None:
        hint = (
            "Call math.find with a mathematical query to search installed operations."
        )
        error_response = {
            "kind": "error",
            "error": {
                "code": "UNKNOWN_OPERATION",
                "stage": "operation_resolution",
                "message": f"Unknown operation: {operation_id}",
                "hint": hint,
            },
        }
        return OperationFindResponse.model_validate(error_response)
    return OperationFindResponse(
        OperationInspectionResult(kind="operation", operation=descriptor)
    )


def math_run(
    operation_id: OperationId,
    payload: dict[str, Any],
    *,
    ctx: Context[AppState, Any],
) -> OperationResult:
    """Run one math tool. Role comes from the tool ID."""
    _authorize(ctx)
    catalog = _catalog(ctx)
    cancellation = _request_cancellation(ctx)
    if cancellation.is_set():
        raise ToolError("operation cancelled before execution")
    try:
        started = time.monotonic()
        # MCP runs synchronous tools in a worker thread.  Its request event
        # is polled by the shared external-process runner, which kills and
        # reaps only an operation's owned child tree.  Bind it before
        # preparation because owner admission may itself use a child worker.
        with bounded_process_cancellation(cancellation), request_execution(started):
            try:
                prepared = _prepare_operation(operation_id, payload, catalog)
            except _OperationResolutionError as exc:
                raise ToolError(str(exc)) from exc
            if cancellation.is_set():
                raise ToolError("operation cancelled before execution")
            return _invoke_prepared_operation(prepared, started=started)
    except (OperationRequestValidationError, OperationDomainValidationError) as exc:
        errors = _bounded_validation_issues(exc.errors())
        data = OperationInvalidRequestData(
            operation_id=operation_id,
            errors=errors,
        )
        raise MCPError(
            code=INVALID_PARAMS,
            message="operation payload failed validation",
            data=data.model_dump(mode="json"),
        ) from exc
    except OperationExecutionTimeoutError as exc:
        raise ToolError("operation execution deadline expired") from exc
    except (MCPError, ToolError):
        raise
    except Exception as exc:
        # Keep backend details inside the owner while guaranteeing the SDK
        # receives a bounded tool error instead of an unhandled worker failure.
        raise ToolError("operation execution failed") from exc


def _request_cancellation(ctx: Context[AppState, Any]) -> _CancellationSignal:
    """Return MCP 2.1's request signal through its only available SDK seam."""

    return ctx.request_context.session._request_outbound.cancel_requested


def _bounded_validation_issues(
    errors: Sequence[Mapping[str, Any]],
) -> tuple[OperationValidationIssue, ...]:
    """Build bounded field diagnostics without reflecting raw caller input."""

    issues: list[OperationValidationIssue] = []
    for error in errors[:_MAX_VALIDATION_ERRORS]:
        issues.append(
            OperationValidationIssue(
                location=_bounded_validation_location(error["loc"]),
                code=str(error["type"]),
                message=_bounded_validation_message(error["msg"]),
            )
        )
    return tuple(issues)


def _bounded_validation_location(value: Any) -> tuple[str | int, ...]:
    """Sanitize caller-controlled Pydantic locations for recovery output."""

    location: list[str | int] = []
    for component in value:
        if isinstance(component, str):
            location.append(_bounded_text(component, _MAX_VALIDATION_LOCATION_LENGTH))
        elif type(component) is int:
            location.append(component)
        if len(location) == _MAX_VALIDATION_LOCATION_COMPONENTS:
            break
    return tuple(location)


def _bounded_validation_message(value: Any) -> str:
    """Keep caller-influenced Pydantic diagnostics inside the public schema."""

    return _bounded_text(str(value), 1_024)


def _bounded_text(value: str, maximum_length: int) -> str:
    return (
        value if len(value) <= maximum_length else f"{value[: maximum_length - 3]}..."
    )
