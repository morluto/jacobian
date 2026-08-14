"""Transport-neutral MCP protocol projection shared by local and remote hosts."""

from __future__ import annotations

import logging
from typing import Any, cast

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.resources import FunctionResource
from mcp.shared.exceptions import MCPError
from mcp_types import INVALID_PARAMS, CallToolResult, InputRequiredResult
from mcp_types import Tool as MCPTool

from jacobian.adapters.mcp.context import (
    AppState,
    _public_tool_error,
)
from jacobian.adapters.mcp.deployment_identity import DeploymentIdentity
from jacobian.adapters.mcp.guidance import MATH_FIND_DESCRIPTION, MATH_RUN_DESCRIPTION
from jacobian.adapters.mcp.tooling import _tool_annotations
from jacobian.adapters.mcp.tools import math_find, math_run
from jacobian.contracts.operations import OperationCatalogSnapshot

_LOGGER = logging.getLogger(__name__)


class JacobianMCPServer(MCPServer[AppState]):
    """Strict two-tool SDK projection used by every Jacobian host."""

    async def list_tools(self) -> list[MCPTool]:
        tools = await super().list_tools()
        return [
            tool.model_copy(
                update={
                    "input_schema": {
                        **tool.input_schema,
                        "additionalProperties": False,
                    }
                }
            )
            for tool in tools
        ]

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: Any | None = None,
    ) -> CallToolResult | InputRequiredResult:
        tools = {tool.name: tool for tool in await super().list_tools()}
        tool = tools.get(name)
        if tool is not None:
            declared_arguments = set(tool.input_schema.get("properties", {}))
            unknown_arguments = sorted(set(arguments) - declared_arguments)
            if unknown_arguments:
                detail = ValueError(
                    "unknown tool arguments: " + ", ".join(unknown_arguments)
                )
                raise MCPError(
                    code=INVALID_PARAMS,
                    message=_public_tool_error(name, detail),
                ) from detail
            stringified_arguments = sorted(
                argument_name
                for argument_name, value in arguments.items()
                if isinstance(value, str)
                and _schema_accepts_container(
                    tool.input_schema,
                    tool.input_schema.get("properties", {}).get(argument_name, {}),
                )
            )
            if stringified_arguments:
                detail = ValueError(
                    "structured tool arguments must not be JSON strings: "
                    + ", ".join(stringified_arguments)
                )
                raise MCPError(
                    code=INVALID_PARAMS,
                    message=_public_tool_error(name, detail),
                ) from detail
        try:
            return await super().call_tool(name, arguments, context)
        except MCPError:
            raise
        except Exception as exc:
            _LOGGER.warning("MCP tool %s failed", name, exc_info=exc)
            raise ToolError(_public_tool_error(name, exc)) from exc


def _schema_accepts_container(
    root: dict[str, Any],
    schema: dict[str, Any],
) -> bool:
    """Detect SDK arguments whose declared value must remain structured JSON."""

    reference = schema.get("$ref")
    if isinstance(reference, str) and reference.startswith("#/$defs/"):
        resolved = root.get("$defs", {}).get(reference.removeprefix("#/$defs/"), {})
        return isinstance(resolved, dict) and _schema_accepts_container(root, resolved)
    declared_type = schema.get("type")
    if declared_type in {"object", "array"}:
        return True
    return any(
        isinstance(branch, dict) and _schema_accepts_container(root, branch)
        for keyword in ("anyOf", "oneOf")
        for branch in schema.get(keyword, ())
    )


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
        annotations=_tool_annotations(read_only=True, idempotent=True),
        structured_output=True,
    )
    server.add_tool(
        math_run,
        name="math.run",
        title="Run one installed Jacobian math tool",
        description=MATH_RUN_DESCRIPTION,
        annotations=_tool_annotations(),
        structured_output=True,
    )

    def operation_catalog() -> OperationCatalogSnapshot:
        return cast(OperationCatalogSnapshot, state.operation_catalog.snapshot())

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


__all__ = ["JacobianMCPServer", "register_core_projection"]
