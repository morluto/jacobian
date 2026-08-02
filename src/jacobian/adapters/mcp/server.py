"""Thin MCP 2.0.0 adapter over the tested Python runtime."""

from __future__ import annotations

import inspect
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import ExitStack, asynccontextmanager
from functools import wraps
from pathlib import Path
from typing import Any

from mcp.server import MCPServer
from mcp.server.extension import Extension, ResourceBinding, ToolBinding
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.resources import FunctionResource, TextResource
from mcp.shared.exceptions import MCPError
from mcp_types import CallToolResult, TextContent

from jacobian import __version__
from jacobian.adapters.mcp.context import (
    AppState,
    _configured_root,
    _public_tool_error,
    _resource_runtime,
    _start_lean_warmup,
)
from jacobian.adapters.mcp.guidance import (
    CAPABILITY_DESCRIBE_DESCRIPTION,
    CAPABILITY_INVOKE_DESCRIPTION,
    OPERATING_GUIDE,
    SERVER_DESCRIPTION,
    SERVER_INSTRUCTIONS,
)
from jacobian.adapters.mcp.projections import (
    WORKSPACE_OPEN_DESCRIPTION,
    WORKSPACE_QUERY_DESCRIPTION,
    WORKSPACE_WRITE_DESCRIPTION,
    CapabilityProjectionStrategy,
)
from jacobian.adapters.mcp.remote import TenantRuntimeRouter
from jacobian.adapters.mcp.resources import _register_resources_and_prompts
from jacobian.adapters.mcp.tooling import (
    _argument_digest,
    _response_size,
    _tool_annotations,
)
from jacobian.adapters.mcp.tools import (
    capability_describe,
    capability_invoke,
    workspace_open,
    workspace_query,
    workspace_write,
)
from jacobian.capabilities import CapabilityPolicy
from jacobian.references import reference_catalog
from jacobian.runtime import CheckerAuthorityMode, create_runtime
from jacobian.runtime.model import JacobianRuntime

_LOGGER = logging.getLogger(__name__)


class JacobianMCPServer(MCPServer[AppState]):
    """MCP server with SDK-owned static argument validation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        for tool in self._tool_manager.list_tools():
            argument_model = tool.fn_metadata.arg_model
            argument_model.model_config["extra"] = "forbid"
            argument_model.model_rebuild(force=True)
            tool.parameters = argument_model.model_json_schema(by_alias=True)

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: Any | None = None,
    ) -> Any:
        try:
            return await super().call_tool(name, arguments, context)
        except ToolError as exc:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=_public_tool_error(name, exc),
                    )
                ],
                is_error=True,
            )


class JacobianCoreExtension(Extension):
    """Stable Jacobian tools and static resources contributed through MCP v2."""

    identifier = "io.jacobian/core"

    def __init__(
        self,
        runtime: JacobianRuntime | None,
        tenant_router: TenantRuntimeRouter | None,
    ) -> None:
        self._runtime = runtime
        self._tenant_router = tenant_router

    def settings(self) -> dict[str, Any]:
        return {"version": "1"}

    def tools(self) -> tuple[ToolBinding, ...]:
        return (
            ToolBinding(
                _safe_tool_handler("capability.describe", capability_describe),
                kwargs={
                    "name": "capability.describe",
                    "title": "Discover mathematical capabilities",
                    "description": CAPABILITY_DESCRIBE_DESCRIPTION,
                    "annotations": _tool_annotations(read_only=True, idempotent=True),
                    "structured_output": True,
                },
            ),
            ToolBinding(
                _safe_tool_handler("capability.invoke", capability_invoke),
                kwargs={
                    "name": "capability.invoke",
                    "title": "Execute a mathematical capability",
                    "description": CAPABILITY_INVOKE_DESCRIPTION,
                    "annotations": _tool_annotations(),
                    "structured_output": True,
                },
            ),
            ToolBinding(
                _safe_tool_handler("workspace.open", workspace_open),
                kwargs={
                    "name": "workspace.open",
                    "description": WORKSPACE_OPEN_DESCRIPTION,
                    "annotations": _tool_annotations(idempotent=True),
                    "structured_output": False,
                },
            ),
            ToolBinding(
                _safe_tool_handler("workspace.write", workspace_write),
                kwargs={
                    "name": "workspace.write",
                    "description": WORKSPACE_WRITE_DESCRIPTION,
                    "annotations": _tool_annotations(idempotent=True),
                    "structured_output": False,
                },
            ),
            ToolBinding(
                _safe_tool_handler("workspace.query", workspace_query),
                kwargs={
                    "name": "workspace.query",
                    "description": WORKSPACE_QUERY_DESCRIPTION,
                    "annotations": _tool_annotations(read_only=True, idempotent=True),
                    "structured_output": False,
                },
            ),
        )

    def resources(self) -> tuple[ResourceBinding, ...]:
        return (
            ResourceBinding(
                TextResource(
                    uri="jacobian://instructions",
                    name="jacobian-instructions",
                    title="Jacobian operating guide",
                    description=(
                        "Complete guidance for discovering, invoking, and independently "
                        "checking Jacobian mathematical capabilities."
                    ),
                    mime_type="text/markdown",
                    text=OPERATING_GUIDE,
                )
            ),
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
            ResourceBinding(
                FunctionResource.from_function(
                    self._reference_catalog,
                    uri="reference://catalog",
                    name="reference-catalog",
                    description=(
                        "Read installed domain schema, semantics, plugin, and checker IDs."
                    ),
                    mime_type="application/json",
                )
            ),
        )

    async def _capability_catalog(self) -> str:
        with _resource_runtime(self._runtime, self._tenant_router) as active_runtime:
            return json.dumps(
                active_runtime.core.capabilities.catalog().model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
            )

    async def _reference_catalog(self) -> str:
        with _resource_runtime(self._runtime, self._tenant_router) as active_runtime:
            return json.dumps(
                reference_catalog(
                    active_runtime.portfolio.references,
                    graph=active_runtime.portfolio.graph,
                    polytope=active_runtime.services.polytope,
                    polytope_checkers=active_runtime.portfolio.polytope_checkers,
                    polynomial=active_runtime.portfolio.polynomial,
                    universal_algebra=active_runtime.portfolio.universal_algebra,
                    lean=active_runtime.portfolio.lean_checkers,
                ),
                ensure_ascii=False,
                sort_keys=True,
            )

    async def intercept_tool_call(
        self,
        params: Any,
        ctx: Any,
        call_next: Any,
    ) -> Any:
        started = time.monotonic()
        arguments = params.arguments or {}
        argument_digest = _argument_digest(arguments)
        try:
            with ExitStack() as request_resources:
                if self._tenant_router is not None:
                    from mcp.server.auth.middleware.auth_context import (
                        get_access_token,
                    )

                    access_token = get_access_token()
                    subject = access_token.subject if access_token is not None else None
                    active_runtime = request_resources.enter_context(
                        self._tenant_router.lease_for(subject)
                    )
                    _start_lean_warmup(active_runtime)
                # MCP 2.0.0 validates declared parameter values through the generated
                # Pydantic model, but that model intentionally ignores unknown keys.
                # Keep this narrow adapter check so the public tool boundary remains
                # closed; domain-selected capability payloads are validated later by
                # Jacobian's descriptor contract.
                binding = next(
                    binding
                    for binding in self.tools()
                    if binding.kwargs["name"] == params.name
                )
                accepted_arguments = {
                    name
                    for name in inspect.signature(binding.fn).parameters
                    if name != "ctx"
                }
                unknown_arguments = sorted(set(arguments) - accepted_arguments)
                if unknown_arguments:
                    raise ValueError(
                        "unknown tool arguments: " + ", ".join(unknown_arguments)
                    )
                result = await call_next(ctx)
        except MCPError:
            _log_tool_call(params.name, started, argument_digest, status="error")
            raise
        except Exception as exc:
            _LOGGER.warning("MCP tool %s failed", params.name, exc_info=exc)
            result = CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=_public_tool_error(params.name, exc),
                    )
                ],
                is_error=True,
            )
            _log_tool_call(
                params.name,
                started,
                argument_digest,
                status="error",
                result=result,
            )
            return result
        _log_tool_call(
            params.name,
            started,
            argument_digest,
            status="success",
            result=result,
        )
        return result


def _safe_tool_handler(tool_name: str, handler: Any) -> Any:
    """Translate internal failures at the handler boundary before SDK rendering."""

    @wraps(handler)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            return await handler(*args, **kwargs)
        except MCPError:
            raise
        except Exception as exc:
            _LOGGER.warning("MCP tool %s failed", tool_name, exc_info=exc)
            raise ToolError(_public_tool_error(tool_name, exc)) from exc

    return wrapped


def _log_tool_call(
    name: str,
    started: float,
    argument_digest: str,
    *,
    status: str,
    result: Any | None = None,
) -> None:
    _LOGGER.info(
        "MCP tool call tool=%s status=%s duration_ms=%.3f "
        "response_bytes=%d argument_digest=%s",
        name,
        status,
        (time.monotonic() - started) * 1000,
        0 if result is None else _response_size(result),
        argument_digest,
    )


def _selected_checker_authority(
    authority: CheckerAuthorityMode | None,
) -> CheckerAuthorityMode:
    if authority is not None:
        return authority
    from jacobian.runtime import CheckerAuthorityMode

    return CheckerAuthorityMode.INSTALL_BUNDLED


@asynccontextmanager
async def _runtime_lifespan(
    _server: Any,
    *,
    runtime: JacobianRuntime | None,
    tenant_router: TenantRuntimeRouter | None,
    projection_strategy: CapabilityProjectionStrategy,
) -> AsyncIterator[AppState]:
    if runtime is not None:
        _start_lean_warmup(runtime)
    try:
        yield AppState(
            runtime=runtime,
            tenant_router=tenant_router,
            projection_strategy=projection_strategy,
        )
    finally:
        if runtime is not None:
            runtime.close()
        if tenant_router is not None:
            tenant_router.close()


def create_server(
    state_dir: str | Path | None = None,
    *,
    checker_authority: CheckerAuthorityMode | None = None,
    tenant_isolation: bool = False,
    allow_anonymous: bool = False,
    anonymous_tenant_id: str = "anonymous",
    token_verifier: Any | None = None,
    auth: Any | None = None,
    capability_adapter_entrypoints: tuple[str, ...] = (),
    capability_exclusions: frozenset[str] = frozenset(),
    capability_policy: CapabilityPolicy | None = None,
    max_tenant_runtimes: int | None = None,
    tenant_idle_timeout_seconds: float | None = None,
    _projection_strategy: CapabilityProjectionStrategy = (
        "COMPACT_URI_TEXT_RESOURCE_LINK"
    ),
) -> MCPServer[AppState]:
    """Create a local or tenant-routed adapter over a Jacobian runtime."""

    if tenant_isolation and capability_exclusions:
        raise ValueError("capability exclusions are supported only by local evaluation")
    if _projection_strategy not in {
        "FULL_INLINE",
        "COMPACT_URI_TEXT",
        "COMPACT_URI_TEXT_RESOURCE_LINK",
    }:
        raise ValueError("unsupported internal MCP projection strategy")

    # Keep ``--help`` and ``--version`` independent of the MCP runtime's
    # heavier imports and shutdown hooks.
    from mcp.server.mcpserver import Context

    from jacobian.adapters.mcp.remote import (
        DEFAULT_MAX_TENANT_RUNTIMES,
        DEFAULT_TENANT_IDLE_TIMEOUT_SECONDS,
        TenantRuntimeRouter,
    )
    from jacobian.runtime.model import JacobianRuntime

    globals().update(
        {
            "Context": Context,
            "JacobianRuntime": JacobianRuntime,
        }
    )

    selected_authority = _selected_checker_authority(checker_authority)
    configured_root = _configured_root(state_dir)
    runtime = (
        None
        if tenant_isolation
        else create_runtime(
            configured_root,
            checker_authority=selected_authority,
            capability_adapter_entrypoints=capability_adapter_entrypoints,
            capability_exclusions=capability_exclusions,
            capability_policy=capability_policy,
        )
    )
    tenant_router = (
        TenantRuntimeRouter(
            configured_root,
            checker_authority=selected_authority,
            allow_anonymous=allow_anonymous,
            anonymous_tenant_id=anonymous_tenant_id,
            capability_adapter_entrypoints=capability_adapter_entrypoints,
            capability_policy=capability_policy,
            max_tenant_runtimes=(
                DEFAULT_MAX_TENANT_RUNTIMES
                if max_tenant_runtimes is None
                else max_tenant_runtimes
            ),
            idle_timeout_seconds=(
                DEFAULT_TENANT_IDLE_TIMEOUT_SECONDS
                if tenant_idle_timeout_seconds is None
                else tenant_idle_timeout_seconds
            ),
        )
        if tenant_isolation
        else None
    )

    @asynccontextmanager
    async def lifespan(server: MCPServer[AppState]) -> AsyncIterator[AppState]:
        async with _runtime_lifespan(
            server,
            runtime=runtime,
            tenant_router=tenant_router,
            projection_strategy=_projection_strategy,
        ) as state:
            yield state

    server: MCPServer[AppState] = JacobianMCPServer(
        name="jacobian",
        title="Jacobian Mathematical Workbench",
        description=SERVER_DESCRIPTION,
        instructions=SERVER_INSTRUCTIONS,
        version=__version__,
        lifespan=lifespan,
        token_verifier=token_verifier,
        auth=auth,
        extensions=[JacobianCoreExtension(runtime, tenant_router)],
    )

    _register_resources_and_prompts(server, runtime, tenant_router)
    return server


if __name__ == "__main__":
    from jacobian.adapters.mcp.cli import main

    main()
