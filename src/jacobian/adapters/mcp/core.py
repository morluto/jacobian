"""Transport-neutral MCP protocol projection shared by local and remote hosts."""

from __future__ import annotations

from mcp.server import MCPServer
from mcp.server.mcpserver.resources import FunctionResource
from mcp.types import ToolAnnotations

from jacobian.adapters.mcp.context import AppState
from jacobian.adapters.mcp.deployment_identity import DeploymentIdentity
from jacobian.adapters.mcp.guidance import MATH_FIND_DESCRIPTION, MATH_RUN_DESCRIPTION
from jacobian.adapters.mcp.tools import math_find, math_run
from jacobian.contracts.operations import OperationCatalogSnapshot


def register_core_projection(
    server: MCPServer[AppState],
    state: AppState,
    deployment_identity: DeploymentIdentity | None = None,
) -> None:
    """Register Jacobian's fixed tools and static resources directly with the SDK."""

    server.add_tool(
        math_find,
        name="math.find",
        title="Search installed Jacobian math tools",
        description=MATH_FIND_DESCRIPTION,
        annotations=ToolAnnotations(
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        ),
        structured_output=True,
    )
    server.add_tool(
        math_run,
        name="math.run",
        title="Run one installed Jacobian math tool",
        description=MATH_RUN_DESCRIPTION,
        annotations=ToolAnnotations(
            read_only_hint=False,
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
                "Installed model-facing operations, supported lanes, and compact schemas."
            ),
            mime_type="application/json",
        )
    )

    if deployment_identity is not None:

        async def managed_deployment_identity() -> DeploymentIdentity:
            return deployment_identity

        server.add_resource(
            FunctionResource.from_function(
                managed_deployment_identity,
                uri="deployment://identity",
                name="deployment-identity",
                description=(
                    "Immutable Git revision and package version for this managed "
                    "Jacobian release."
                ),
                mime_type="application/json",
            )
        )


__all__ = ["register_core_projection"]
