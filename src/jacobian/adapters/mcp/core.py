"""Transport-neutral MCP protocol projection shared by local and remote hosts."""

from __future__ import annotations

import logging
import time
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
from jacobian.adapters.mcp.tooling import (
    AgentRecoveryError,
    _argument_digest,
    _request_id_digest,
    _request_trace_digest,
    _response_size,
    _tool_annotations,
)
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
        try:
            return await super().call_tool(name, arguments, context)
        except MCPError:
            raise
        except Exception as exc:
            _LOGGER.warning("MCP tool %s failed", name, exc_info=exc)
            raise ToolError(_public_tool_error(name, exc)) from exc


class JacobianCoreExtension(Extension):
    """Stable Jacobian tools and static resources contributed through MCP v2."""

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
        started = time.monotonic()
        arguments = params.arguments or {}
        argument_digest = _argument_digest(arguments)
        request_digest = _request_id_digest(ctx)
        trace_digest, trace_source = _request_trace_digest(ctx)
        try:
            state = ctx.lifespan_context
            if not isinstance(state, AppState):
                raise AgentRecoveryError(
                    "Jacobian is unavailable for this request. Retry once; if it "
                    "fails again, inspect the local Jacobian log."
                )
            with _runtime_scope(state):
                result = await call_next(ctx)
        except MCPError:
            _log_tool_call(
                params.name,
                started,
                argument_digest,
                request_digest=request_digest,
                trace_digest=trace_digest,
                trace_source=trace_source,
                status="error",
            )
            raise
        except Exception:
            _log_tool_call(
                params.name,
                started,
                argument_digest,
                request_digest=request_digest,
                trace_digest=trace_digest,
                trace_source=trace_source,
                status="error",
            )
            raise
        _log_tool_call(
            params.name,
            started,
            argument_digest,
            request_digest=request_digest,
            trace_digest=trace_digest,
            trace_source=trace_source,
            status=("error" if getattr(result, "is_error", False) else "success"),
            result=result,
        )
        return result


def _log_tool_call(
    name: str,
    started: float,
    argument_digest: str,
    *,
    request_digest: str,
    trace_digest: str,
    trace_source: str,
    status: str,
    result: Any | None = None,
) -> None:
    _LOGGER.info(
        "MCP tool call tool=%s status=%s request_digest=%s "
        "trace_digest=%s trace_source=%s duration_ms=%.3f "
        "response_bytes=%d argument_digest=%s",
        name,
        status,
        request_digest,
        trace_digest,
        trace_source,
        (time.monotonic() - started) * 1000,
        0 if result is None else _response_size(result),
        argument_digest,
    )


__all__ = ["JacobianCoreExtension", "JacobianMCPServer"]
