"""MCP tool handlers for the operation surface."""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import Context

from jacobian.adapters.mcp.context import AppState, _catalog, _runtime
from jacobian.adapters.mcp.projections import (
    _operation_discovery_response,
)
from jacobian.adapters.mcp.tooling import (
    _invoke_operation_attempt,
)
from jacobian.contracts.operation_find import (
    OperationFindRequest,
    OperationFindResponse,
    OperationInspectionResult,
    OperationSearchRequest,
)
from jacobian.contracts.operations import (
    OperationDiscoveryRequest,
    OperationId,
    OperationResult,
)
from jacobian.runtime.model import JacobianRuntime


def _find_result(response: dict[str, Any]) -> OperationFindResponse:
    if "error" in response and response.get("kind") != "error":
        response = {"kind": "error", **response}
    return OperationFindResponse.model_validate(response)


def _unknown_operation_context(
    runtime: Any,
    operation_id: OperationId,
) -> dict[str, Any]:
    """Return bounded SDK-facing recovery without embedding the full catalog."""

    operations = getattr(getattr(runtime, "core", None), "operations", runtime)
    discovered = operations.search(
        OperationDiscoveryRequest(query=operation_id, limit=5)
    )
    return {
        "nearby_operation_ids": [match.operation_id for match in discovered.matches],
        "available_recovery_paths": [
            {
                "action": "search",
                "tool": "math.find",
                "arguments": {
                    "request": {
                        "op": "search",
                        "query": operation_id,
                        "limit": 5,
                    }
                },
            },
            {
                "action": "inspect_catalog",
                "resource_uri": "operation://catalog",
            },
        ],
    }


def _bounded_run_result(
    runtime: JacobianRuntime,
    result: OperationResult,
) -> OperationResult:
    """Keep unknown-operation recovery small in structured MCP output."""

    if not (result.diagnostics and result.diagnostics[0].code == "UNKNOWN_OPERATION"):
        return result
    payload = result.model_dump(mode="json")
    output = payload["output"]
    output.update(_unknown_operation_context(runtime, result.operation_id))
    payload["output"] = output
    return OperationResult.model_validate(payload)


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
                **_unknown_operation_context(
                    active_catalog,
                    operation_id,
                ),
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
    with _runtime(ctx, operation_id) as active_runtime:
        result = _invoke_operation_attempt(
            active_runtime,
            operation_id=operation_id,
            payload=payload,
        )
        return _bounded_run_result(active_runtime, result)
