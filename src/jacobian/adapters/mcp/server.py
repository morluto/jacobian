"""Local single-runtime MCP host."""

from __future__ import annotations

import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from mcp.server import MCPServer

from jacobian import __version__
from jacobian.adapters.mcp.context import (
    AppState,
    RuntimeAccess,
)
from jacobian.adapters.mcp.core import register_core_projection
from jacobian.adapters.mcp.deployment_identity import (
    DeploymentIdentity,
    load_deployment_identity,
)
from jacobian.adapters.mcp.guidance import SERVER_DESCRIPTION, SERVER_INSTRUCTIONS
from jacobian.adapters.mcp.lifecycle import (
    runtime_lifespan,
)
from jacobian.operation_visibility import OperationVisibilityPolicy
from jacobian.runtime.execution import (
    create_inline_serving_runtime,
)
from jacobian.runtime.model import JacobianRuntime
from jacobian.serving_catalog import ServingCatalog


def create_server(
    *, operation_policy: OperationVisibilityPolicy | None = None
) -> MCPServer[AppState]:
    """Create the stateless local host over the immutable operation library.

    Built-in math neither reads nor creates server-owned state.
    """

    policy = operation_policy or OperationVisibilityPolicy()
    catalog = ServingCatalog.open(policy=policy)
    owner = _LazyLocalRuntime(catalog)
    state = AppState(
        acquire_runtime=owner.acquire,
        operation_catalog=catalog,
    )
    return _build_server(
        state=state,
        close_owner=owner.close,
        deployment_identity=load_deployment_identity(),
    )


class _LazyLocalRuntime:
    def __init__(
        self,
        catalog: ServingCatalog,
    ) -> None:
        self.catalog = catalog
        self._inline_runtime: JacobianRuntime | None = None
        self._lock = threading.Lock()

    def acquire(self, operation_id: str | None = None) -> RuntimeAccess:
        del operation_id
        with self._lock:
            return RuntimeAccess(self._ensure_inline_runtime())

    def _ensure_inline_runtime(self) -> JacobianRuntime:
        if self._inline_runtime is None:
            self._inline_runtime = create_inline_serving_runtime(self.catalog)
        return self._inline_runtime

    def close(self) -> None:
        with self._lock:
            runtime = self._inline_runtime
            self._inline_runtime = None
        if runtime is not None:
            runtime.close()


def create_server_from_runtime(
    runtime: JacobianRuntime,
    *,
    close_owner: Callable[[], None],
    start_owner: Callable[[], None] | None = None,
) -> MCPServer[AppState]:
    """Register the local MCP projection over an explicitly owned runtime.

    The caller supplies lifecycle actions so deployment bootstrap and focused
    compositions exercise identical SDK registration without obscuring which
    layer owns runtime startup and teardown.
    """

    state = AppState(
        acquire_runtime=lambda _operation_id: RuntimeAccess(runtime),
        operation_catalog=runtime.operations,
    )
    return _build_server(
        state=state,
        close_owner=close_owner,
        start_owner=start_owner,
        deployment_identity=load_deployment_identity(),
    )


def _build_server(
    *,
    state: AppState,
    close_owner: Callable[[], None],
    start_owner: Callable[[], None] | None = None,
    deployment_identity: DeploymentIdentity | None = None,
    token_verifier: Any | None = None,
    auth: Any | None = None,
) -> MCPServer[AppState]:
    """Register Jacobian's fixed MCP projection over one runtime owner."""

    @asynccontextmanager
    async def lifespan(server: MCPServer[AppState]) -> AsyncIterator[AppState]:
        async with runtime_lifespan(
            server,
            state=state,
            close_owner=close_owner,
            start_owner=start_owner,
        ) as active_state:
            yield active_state

    server: MCPServer[AppState] = MCPServer(
        name="jacobian",
        title="Jacobian Mathematical Workbench",
        description=SERVER_DESCRIPTION,
        instructions=SERVER_INSTRUCTIONS,
        version=__version__,
        lifespan=lifespan,
        token_verifier=token_verifier,
        auth=auth,
    )
    register_core_projection(server, state, deployment_identity)
    return server


if __name__ == "__main__":
    from jacobian.adapters.mcp.cli import main

    main()


__all__ = ["create_server", "create_server_from_runtime"]
