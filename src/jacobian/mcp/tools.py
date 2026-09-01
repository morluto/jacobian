"""MCP tool handlers for the operation surface."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp.shared.exceptions import MCPError
from mcp.types import INVALID_PARAMS

from jacobian._execution import (
    OperationExecutionCancelledError,
    RequestCancellationSignal,
)
from jacobian.catalog.models import OperationId, OperationResult
from jacobian.dispatch import (
    OperationDomainValidationError,
    OperationExecutionTimeoutError,
    OperationRequestValidationError,
    _OperationResolutionError,
    execute_operation,
)
from jacobian.mcp.models import (
    OperationDiscoveryError,
    OperationDiscoveryErrorDetail,
    OperationFindRequest,
    OperationFindResponse,
    OperationInspectionResult,
    OperationInvalidRequestData,
    OperationMatchRequest,
    OperationValidationIssue,
)
from jacobian.mcp.projections import _operation_match_response
from jacobian.mcp.runtime import (
    AppState,
    _authorize,
    _catalog,
)

_MAX_VALIDATION_ERRORS = 64
_MAX_VALIDATION_LOCATION_COMPONENTS = 32
_MAX_VALIDATION_LOCATION_LENGTH = 128


def math_find(
    request: OperationFindRequest,
    *,
    ctx: Context[AppState, Any],
) -> OperationFindResponse:
    active_catalog = _catalog(ctx)
    if isinstance(request, OperationMatchRequest):
        match_response = _operation_match_response(
            active_catalog,
            need=request.need,
            namespace=request.namespace,
            limit=request.limit,
            cursor=request.cursor,
        )
        return OperationFindResponse(root=match_response)

    operation_id = request.operation_id
    descriptor = active_catalog.inspect(operation_id)
    if descriptor is None:
        hint = (
            "Call math.find with a local mathematical need to match installed "
            "operations."
        )
        return OperationFindResponse(
            root=OperationDiscoveryError(
                kind="error",
                error=OperationDiscoveryErrorDetail(
                    code="UNKNOWN_OPERATION",
                    stage="operation_resolution",
                    message=f"Unknown operation: {operation_id}",
                    hint=hint,
                ),
            )
        )
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
    try:
        try:
            return execute_operation(
                operation_id,
                payload,
                catalog,
                projector=lambda selected_id, result, started: OperationResult(
                    operation_id=selected_id,
                    runtime_ms=max(0, round((time.monotonic() - started) * 1000)),
                    output=result.model_dump(mode="json"),
                ),
                cancellation_signal=cancellation,
            )
        except _OperationResolutionError as exc:
            raise ToolError(str(exc)) from exc
    except (OperationRequestValidationError, OperationDomainValidationError) as exc:
        raise _invalid_request_error(operation_id, exc) from exc
    except OperationExecutionTimeoutError as exc:
        raise ToolError("operation execution deadline expired") from exc
    except OperationExecutionCancelledError as exc:
        raise ToolError("operation cancelled") from exc
    except (MCPError, ToolError):
        raise
    except Exception as exc:
        # Keep backend details inside the owner while guaranteeing the SDK
        # receives a bounded tool error instead of an unhandled worker failure.
        raise ToolError("operation execution failed") from exc


def _invalid_request_error(
    operation_id: OperationId,
    error: OperationRequestValidationError | OperationDomainValidationError,
) -> MCPError:
    """Project one owner-bound rejection without reflecting caller values."""

    data = OperationInvalidRequestData(
        operation_id=operation_id,
        errors=_bounded_validation_issues(error.errors()),
    )
    return MCPError(
        code=INVALID_PARAMS,
        message="operation payload failed validation",
        data=data.model_dump(mode="json"),
    )


def _request_cancellation(ctx: Context[AppState, Any]) -> RequestCancellationSignal:
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
