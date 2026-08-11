"""Thin MCP 2.0.0 adapter over the tested Python runtime."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from mcp.server import MCPServer
from mcp.server.extension import Extension, ResourceBinding, ToolBinding
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.resources import FunctionResource, TextResource
from mcp.shared.exceptions import MCPError
from mcp_types import INVALID_PARAMS, CallToolResult, InputRequiredResult
from mcp_types import Tool as MCPTool

from jacobian import __version__
from jacobian.adapters.mcp.context import (
    AppState,
    _configured_root,
    _public_tool_error,
    _runtime_scope,
    _start_lean_warmup,
    _static_resource_runtime,
)
from jacobian.adapters.mcp.guidance import (
    MATH_FIND_DESCRIPTION,
    MATH_RUN_DESCRIPTION,
    OPERATING_GUIDE,
    SERVER_DESCRIPTION,
    SERVER_INSTRUCTIONS,
)
from jacobian.adapters.mcp.remote import (
    DEFAULT_MAX_TENANT_RUNTIMES,
    DEFAULT_TENANT_IDLE_TIMEOUT_SECONDS,
    TenantRuntimeRouter,
)
from jacobian.adapters.mcp.resources import _register_resources_and_prompts
from jacobian.adapters.mcp.tooling import (
    AgentRecoveryError,
    MCPBlockingWorkerRegistry,
    MCPBlockingWorkerShutdownError,
    _argument_digest,
    _request_id_digest,
    _request_trace_digest,
    _response_size,
    _tool_annotations,
)
from jacobian.adapters.mcp.tools import (
    capability_describe,
    capability_invoke,
)
from jacobian.capability_service import CapabilityPolicy
from jacobian.contracts.capabilities import CapabilityCatalog
from jacobian.references import reference_catalog
from jacobian.runtime import CheckerAuthorityMode, create_runtime
from jacobian.runtime.model import JacobianRuntime

_LOGGER = logging.getLogger(__name__)


class JacobianMCPServer(MCPServer[AppState]):
    """Public-API compatibility seam for strict MCP 2.0.0 tool inputs.

    The pinned SDK owns argument model generation and validation, but its generated
    models silently ignore extra keys and expose no strictness setting. This adapter
    closes only that gap through the SDK's public tool listing and invocation APIs;
    declared values and results still use the native SDK validators and converters.
    """

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
        runtime: JacobianRuntime | None,
        tenant_router: TenantRuntimeRouter | None,
    ) -> None:
        self._runtime = runtime
        self._tenant_router = tenant_router

    def settings(self) -> dict[str, Any]:
        return {"version": "2"}

    def tools(self) -> tuple[ToolBinding, ...]:
        return (
            ToolBinding(
                capability_describe,
                kwargs={
                    "name": "math.find",
                    "title": "Find an exact mathematical operation",
                    "description": MATH_FIND_DESCRIPTION,
                    "annotations": _tool_annotations(read_only=True, idempotent=True),
                    "structured_output": True,
                },
            ),
            ToolBinding(
                capability_invoke,
                kwargs={
                    "name": "math.run",
                    "title": "Run a mathematical operation",
                    "description": MATH_RUN_DESCRIPTION,
                    # math.run dispatches the installed portfolio, including
                    # state-changing operations such as experiment.cancel.
                    "annotations": _tool_annotations(destructive=True),
                    "structured_output": True,
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

    async def _capability_catalog(self) -> CapabilityCatalog:
        with _static_resource_runtime(
            self._runtime, self._tenant_router
        ) as active_runtime:
            return active_runtime.core.capabilities.catalog()

    async def _reference_catalog(self) -> dict[str, Any]:
        with _static_resource_runtime(
            self._runtime, self._tenant_router
        ) as active_runtime:
            return reference_catalog(
                active_runtime.portfolio.references,
                graph=active_runtime.portfolio.graph,
                polytope=active_runtime.services.polytope,
                polytope_checkers=active_runtime.portfolio.polytope_checkers,
                polynomial=active_runtime.portfolio.polynomial,
                universal_algebra=active_runtime.portfolio.universal_algebra,
                lean=active_runtime.portfolio.lean_checkers,
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
            # JacobianMCPServer.call_tool is the single public error boundary.
            # Re-raise here so telemetry observes the original failure once.
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
    worker_registry: MCPBlockingWorkerRegistry,
) -> AsyncIterator[AppState]:
    if runtime is not None:
        _start_lean_warmup(runtime)
    try:
        yield AppState(
            runtime=runtime,
            worker_registry=worker_registry,
            tenant_router=tenant_router,
        )
    finally:
        try:
            await worker_registry.close()
        except MCPBlockingWorkerShutdownError as exc:
            # A late thread still has a registry-owned tenant lease.  Closing its
            # runtime now would let it use a torn-down store.  Keep ownership in
            # the registry and close it once the final late worker has released
            # its request scope.
            worker_registry.defer_until_quiescent(
                lambda: _close_runtime_owners(runtime, tenant_router)
            )
            raise exc from None
        _close_runtime_owners(runtime, tenant_router)


def _close_runtime_owners(
    runtime: JacobianRuntime | None,
    tenant_router: TenantRuntimeRouter | None,
) -> None:
    """Close runtime owners once no request worker can still use them."""

    cleanup_failures: list[BaseException] = []
    if runtime is not None:
        try:
            runtime.close()
        except BaseException as exc:
            cleanup_failures.append(exc)
    if tenant_router is not None:
        try:
            tenant_router.close()
        except BaseException as exc:
            cleanup_failures.append(exc)
    if cleanup_failures:
        if len(cleanup_failures) == 1:
            raise cleanup_failures[0]
        raise BaseExceptionGroup(
            "runtime and tenant router cleanup failed", cleanup_failures
        )


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
) -> MCPServer[AppState]:
    """Create a local or tenant-routed adapter over a Jacobian runtime."""

    if tenant_isolation and capability_exclusions:
        raise ValueError("capability exclusions are supported only by local evaluation")

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
    worker_registry = MCPBlockingWorkerRegistry()

    @asynccontextmanager
    async def lifespan(server: MCPServer[AppState]) -> AsyncIterator[AppState]:
        async with _runtime_lifespan(
            server,
            runtime=runtime,
            tenant_router=tenant_router,
            worker_registry=worker_registry,
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
        extensions=[
            JacobianCoreExtension(
                runtime,
                tenant_router,
            )
        ],
    )

    _register_resources_and_prompts(server)
    return server


if __name__ == "__main__":
    from jacobian.adapters.mcp.cli import main

    main()
