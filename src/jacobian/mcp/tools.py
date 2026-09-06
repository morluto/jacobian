"""MCP tool handlers for the operation surface."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
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
from jacobian.backends import BackendUnavailableError, check_backend
from jacobian.catalog.models import (
    OperationId,
    OperationResourceAdmissionError,
    OperationResult,
)
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
    OperationResourceAdmissionData,
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
_FIND_QUERY_HASH_HEX_LENGTH = 16
_FIND_QUERY_LOG_KEY = secrets.token_bytes(32)

logger = logging.getLogger(__name__)


def math_find(
    request: OperationFindRequest,
    *,
    ctx: Context[AppState, Any],
) -> OperationFindResponse:
    active_catalog = _catalog(ctx)
    if isinstance(request, OperationMatchRequest):
        # The process-local HMAC key prevents log readers from recovering a
        # short caller need through an offline digest lookup.
        need_hash = hmac.new(
            _FIND_QUERY_LOG_KEY, request.need.encode("utf-8"), hashlib.sha256
        ).hexdigest()[:_FIND_QUERY_HASH_HEX_LENGTH]
        logger.info("math.find query_hash=%s", need_hash)
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
        OperationInspectionResult(
            kind="operation",
            operation=descriptor,
            backend_availability=tuple(
                check_backend(name) for name in descriptor.runtime_requirements
            ),
        )
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
        raise _execution_tool_error(
            code="OPERATION_TIMEOUT", operation_id=operation_id, stage=exc.stage
        ) from exc
    except OperationExecutionCancelledError as exc:
        raise _execution_tool_error(
            code="OPERATION_CANCELLED", operation_id=operation_id, stage=exc.stage
        ) from exc
    except BackendUnavailableError as exc:
        raise _backend_unavailable_error(operation_id, exc) from exc
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

    issues = _bounded_validation_issues(error.errors())
    if isinstance(error, OperationResourceAdmissionError):
        data: OperationInvalidRequestData | OperationResourceAdmissionData = (
            OperationResourceAdmissionData(operation_id=operation_id, errors=issues)
        )
        message = "operation request exceeds its admitted resource envelope"
    else:
        data = OperationInvalidRequestData(operation_id=operation_id, errors=issues)
        message = "operation payload failed validation"
    return MCPError(
        code=INVALID_PARAMS,
        message=message,
        data=data.model_dump(mode="json"),
    )


def _backend_unavailable_error(
    operation_id: str, error: BackendUnavailableError
) -> ToolError:
    """Expose backend recovery without changing a mathematical result schema."""

    return ToolError(
        json.dumps(
            {
                "code": "BACKEND_UNAVAILABLE",
                "stage": "backend_execution",
                "operation_id": operation_id,
                "backend": error.backend,
                "required_version": error.required_version,
                "hint": error.installation,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def _execution_tool_error(*, code: str, operation_id: str, stage: str) -> ToolError:
    """Project bounded operation context through the SDK's text-only tool error."""

    return ToolError(
        json.dumps(
            {"code": code, "operation_id": operation_id, "stage": stage},
            separators=(",", ":"),
            sort_keys=True,
        )
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
