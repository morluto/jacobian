"""Local single-runtime MCP host."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from mcp.server import MCPServer

from jacobian import __version__
from jacobian.adapters.mcp.context import (
    AppState,
    RuntimeLease,
    _configured_root,
    _start_lean_warmup,
)
from jacobian.adapters.mcp.core import JacobianCoreExtension, JacobianMCPServer
from jacobian.adapters.mcp.deployment_identity import load_deployment_identity
from jacobian.adapters.mcp.guidance import SERVER_DESCRIPTION, SERVER_INSTRUCTIONS
from jacobian.adapters.mcp.lifecycle import (
    runtime_lifespan,
    selected_checker_authority,
)
from jacobian.adapters.mcp.resources import register_resources
from jacobian.adapters.mcp.tooling import MCPBlockingWorkerRegistry
from jacobian.capability_service import CapabilityPolicy
from jacobian.runtime import CheckerAuthorityMode, create_runtime


def create_server(
    state_dir: str | Path | None = None,
    *,
    checker_authority: CheckerAuthorityMode | None = None,
    capability_exclusions: frozenset[str] = frozenset(),
    capability_policy: CapabilityPolicy | None = None,
) -> MCPServer[AppState]:
    """Create one local server owning one mathematical runtime."""

    runtime = create_runtime(
        _configured_root(state_dir),
        checker_authority=selected_checker_authority(checker_authority),
        capability_exclusions=capability_exclusions,
        capability_policy=capability_policy,
    )
    state = AppState(
        acquire_runtime=lambda: RuntimeLease(runtime),
        worker_registry=MCPBlockingWorkerRegistry(),
    )
    return _build_local_server(
        state=state,
        close_owner=runtime.close,
        start_owner=lambda: _start_lean_warmup(runtime),
    )


def _build_local_server(
    *,
    state: AppState,
    close_owner: Callable[[], None],
    start_owner: Callable[[], None] | None = None,
) -> MCPServer[AppState]:
    """Register the fixed MCP projection over one local runtime owner."""

    @asynccontextmanager
    async def lifespan(server: MCPServer[AppState]) -> AsyncIterator[AppState]:
        async with runtime_lifespan(
            server,
            state=state,
            close_owner=close_owner,
            start_owner=start_owner,
        ) as active_state:
            yield active_state

    server: MCPServer[AppState] = JacobianMCPServer(
        name="jacobian",
        title="Jacobian Mathematical Workbench",
        description=SERVER_DESCRIPTION,
        instructions=SERVER_INSTRUCTIONS,
        version=__version__,
        lifespan=lifespan,
        extensions=[JacobianCoreExtension(state, load_deployment_identity())],
    )
    register_resources(server)
    return server


if __name__ == "__main__":
    from jacobian.adapters.mcp.cli import main

    main()


__all__ = ["create_server"]
