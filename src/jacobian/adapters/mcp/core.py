"""Transport-neutral MCP protocol projection shared by local and remote hosts."""

from __future__ import annotations

import logging
from typing import Any

from mcp.server import MCPServer
from mcp.server.extension import Extension, ResourceBinding, ToolBinding
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.resources import FunctionResource
from mcp.shared.exceptions import MCPError
from mcp_types import INVALID_PARAMS, CallToolResult, InputRequiredResult
from mcp_types import Tool as MCPTool

from jacobian.adapters.mcp.context import (
    AppState,
    _public_tool_error,
    _runtime_scope,
    _static_resource_runtime,
)
from jacobian.adapters.mcp.deployment_identity import DeploymentIdentity
from jacobian.adapters.mcp.guidance import MATH_FIND_DESCRIPTION, MATH_RUN_DESCRIPTION
from jacobian.adapters.mcp.tooling import AgentRecoveryError, _tool_annotations
from jacobian.adapters.mcp.tools import capability_describe, capability_invoke
from jacobian.contracts.capabilities import CapabilityCatalog

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


class JacobianCoreExtension(Extension):
    """Advertised, versioned contract for Jacobian's fixed MCP surface.

    This remains an SDK extension deliberately: clients can identify the
    two-tool Jacobian protocol contract through ``io.jacobian/core``.
    """

    identifier = "io.jacobian/core"

    def __init__(
        self,
        state: AppState,
        deployment_identity: DeploymentIdentity | None = None,
    ) -> None:
        self._state = state
        self._deployment_identity = deployment_identity

    def settings(self) -> dict[str, Any]:
        return {"version": "2"}

    def tools(self) -> tuple[ToolBinding, ...]:
        return (
            ToolBinding(
                capability_describe,
                kwargs={
                    "name": "math.find",
                    "title": "Search installed Jacobian math tools",
                    "description": MATH_FIND_DESCRIPTION,
                    "annotations": _tool_annotations(read_only=True, idempotent=True),
                    "structured_output": True,
                },
            ),
            ToolBinding(
                capability_invoke,
                kwargs={
                    "name": "math.run",
                    "title": "Run one installed Jacobian math tool",
                    "description": MATH_RUN_DESCRIPTION,
                    "annotations": _tool_annotations(),
                    "structured_output": True,
                },
            ),
        )

    def resources(self) -> tuple[ResourceBinding, ...]:
        bindings = [
            ResourceBinding(
                FunctionResource.from_function(
                    self._capability_catalog,
                    uri="capability://catalog",
                    name="capability-catalog",
                    description=(
                        "Installed model-facing operations, supported lanes, and "
                        "compact schemas."
                    ),
                    mime_type="application/json",
                )
            ),
        ]
        if self._deployment_identity is not None:
            bindings.append(
                ResourceBinding(
                    FunctionResource.from_function(
                        self._managed_deployment_identity,
                        uri="deployment://identity",
                        name="deployment-identity",
                        description=(
                            "Immutable Git revision and package version for this "
                            "managed Jacobian release."
                        ),
                        mime_type="application/json",
                    )
                )
            )
        return tuple(bindings)

    async def _capability_catalog(self) -> CapabilityCatalog:
        with _static_resource_runtime(self._state) as active_runtime:
            return active_runtime.core.capabilities.catalog()

    async def _managed_deployment_identity(self) -> DeploymentIdentity:
        identity = self._deployment_identity
        if identity is None:
            raise RuntimeError("managed deployment identity is unavailable")
        return identity

    async def intercept_tool_call(
        self,
        params: Any,
        ctx: Any,
        call_next: Any,
    ) -> Any:
        state = ctx.lifespan_context
        if not isinstance(state, AppState):
            raise AgentRecoveryError(
                "Jacobian is unavailable for this request. Retry once; if it "
                "fails again, inspect the local Jacobian log."
            )
        # The SDK's OpenTelemetry middleware owns protocol tracing, duration,
        # status, and trace propagation. This interceptor owns only Jacobian's
        # runtime lease and blocking-worker request scope.
        with _runtime_scope(state):
            return await call_next(ctx)


__all__ = ["JacobianCoreExtension", "JacobianMCPServer"]
