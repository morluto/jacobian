"""Transport-neutral MCP protocol projection shared by local and remote hosts."""

from __future__ import annotations

from mcp.server import MCPServer
from mcp.server.mcpserver.resources import FunctionResource
from mcp.types import ToolAnnotations

from jacobian.catalog.models import OperationCatalogSnapshot
from jacobian.mcp.delivery import (
    DeliveredFindResult,
    DeliveredRunResult,
    with_delivery_schema,
)
from jacobian.mcp.guidance import MATH_FIND_DESCRIPTION, MATH_RUN_DESCRIPTION
from jacobian.mcp.runtime import AppState
from jacobian.mcp.tools import math_find, math_run


def register_core_projection(
    server: MCPServer[AppState],
    state: AppState,
) -> None:
    """Register Jacobian's fixed tools and static resources directly with the SDK."""

    delivery = state.result_delivery
    if delivery is not None and state.authorize is not None:
        raise ValueError(
            "temporary result delivery is currently supported by the local host only"
        )
    delivery_guidance = (
        " With a configured local delivery budget, a large response has kind "
        "'retained_result' and an MCP resource link. Read the complete JSON from "
        "its uri before using it; the preview is incomplete and the reference expires."
        " Oversized errors retain isError and kind 'retained_error'; retrieve their "
        "complete tool_result resource for the original diagnostic."
        if delivery is not None
        else ""
    )
    find_handler = (
        math_find
        if delivery is None
        else with_delivery_schema(math_find, DeliveredFindResult)
    )
    run_handler = (
        math_run
        if delivery is None
        else with_delivery_schema(math_run, DeliveredRunResult)
    )

    server.add_tool(
        find_handler,
        name="math.find",
        title="Search installed Jacobian math tools",
        description=MATH_FIND_DESCRIPTION + delivery_guidance,
        annotations=ToolAnnotations(
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        ),
        structured_output=True,
    )
    server.add_tool(
        run_handler,
        name="math.run",
        title="Run one installed Jacobian math tool",
        description=MATH_RUN_DESCRIPTION + delivery_guidance,
        annotations=ToolAnnotations(
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=False,
            open_world_hint=False,
        ),
        structured_output=True,
    )

    def operation_catalog() -> OperationCatalogSnapshot:
        return state.operation_catalog.snapshot()

    server.add_resource(
        FunctionResource.from_function(
            operation_catalog,
            uri="operation://catalog",
            name="operation-catalog",
            description=(
                "Large exact bulk export of every installed operation, schema, and "
                "example. Use math.find for ordinary operation discovery and "
                "inspection. If matching fails, refine or paginate math.find; read "
                "this resource only when the task requires the complete inventory."
            ),
            mime_type="application/json",
        )
    )

    if delivery is not None:

        @server.resource("jacobian-result://{token}", mime_type="application/json")
        async def retained_result(token: str) -> str:
            return delivery.read(token)


__all__ = ["register_core_projection"]
