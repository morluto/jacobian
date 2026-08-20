"""MCP tool handlers for the operation surface."""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import Context
from mcp.shared.exceptions import MCPError
from mcp_types import INVALID_PARAMS
from pydantic import ValidationError

from jacobian.catalog.models import OperationId, OperationResult
from jacobian.dispatch import invoke_operation
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
            domain=request.domain,
            limit=request.limit,
            cursor=request.cursor,
        )
        return OperationFindResponse.model_validate(discovery_response)
    if isinstance(request, OperationBrowseRequest):
        browse_response = _operation_browse_response(
            active_catalog,
            domain=request.domain,
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
    try:
        return invoke_operation(
            operation_id,
            payload,
            catalog,
        )
    except ValidationError as exc:
        errors = tuple(
            OperationValidationIssue(
                location=tuple(
                    item for item in error["loc"] if isinstance(item, (str, int))
                ),
                code=str(error["type"]),
                message=str(error["msg"]),
                input=_recoverable_error_input(error.get("input")),
            )
            for error in exc.errors(
                include_url=False,
                include_context=False,
                include_input=True,
            )[:64]
        )
        data = OperationInvalidRequestData(
            operation_id=operation_id,
            errors=errors,
        )
        raise MCPError(
            code=INVALID_PARAMS,
            message="operation payload failed validation",
            data=data.model_dump(mode="json"),
        ) from exc


def _recoverable_error_input(value: Any) -> Any | None:
    """Return bounded JSON error input without rendering it into error text."""

    from jacobian.canonical import CanonicalizationError, encode_strict_json

    try:
        encoded = encode_strict_json(value)
    except CanonicalizationError:
        return None
    return value if len(encoded) <= 2_048 else None
