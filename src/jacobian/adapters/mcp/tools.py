"""MCP tool handlers for the operation surface."""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import Context

from jacobian.adapters.mcp.context import (
    AppState,
    _authorize,
    _catalog,
)
from jacobian.adapters.mcp.projections import (
    _operation_discovery_response,
)
from jacobian.contracts.operation_find import (
    OperationFindRequest,
    OperationFindResponse,
    OperationInspectionResult,
    OperationSearchRequest,
)
from jacobian.contracts.operations import OperationId, OperationResult
from jacobian.operation_dispatcher import invoke_operation


def _find_result(response: dict[str, Any]) -> OperationFindResponse:
    if "error" in response and response.get("kind") != "error":
        response = {"kind": "error", **response}
    return OperationFindResponse.model_validate(response)


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
        return _find_result(discovery_response)
    operation_id = request.operation_id
    descriptor = active_catalog.inspect(operation_id)
    if descriptor is None:
        hint = (
            "Call math.find with a mathematical query to search installed operations."
        )
        error_response = {
            "error": {
                "code": "UNKNOWN_OPERATION",
                "stage": "operation_resolution",
                "message": f"Unknown operation: {operation_id}",
                "hint": hint,
            }
        }
        return _find_result(error_response)
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
    return invoke_operation(
        operation_id,
        payload,
        catalog,
    )
