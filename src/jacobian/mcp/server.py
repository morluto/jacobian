"""Local MCP host over stateless mathematical operations."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

import anyio
from mcp.server import MCPServer
from mcp.server.mcpserver.tools import Tool

from jacobian import __version__
from jacobian.catalog.catalog import Catalog
from jacobian.mcp.delivery import ResultDelivery
from jacobian.mcp.guidance import SERVER_DESCRIPTION, SERVER_INSTRUCTIONS
from jacobian.mcp.protocol import register_core_projection
from jacobian.mcp.runtime import AppState


def create_server(
    *, result_delivery: ResultDelivery | None = None
) -> MCPServer[AppState]:
    """Create a local host with an optional bounded delivery policy.

    Built-in math neither reads nor creates server-owned state.
    """

    catalog = Catalog.open()
    state = AppState(
        operation_catalog=catalog,
        result_delivery=result_delivery,
    )
    return _build_server(state=state)


def _build_server(
    *,
    state: AppState,
    token_verifier: Any | None = None,
    auth: Any | None = None,
    evaluation_tools: Sequence[Tool] = (),
) -> MCPServer[AppState]:
    """Register Jacobian's fixed MCP projection over one immutable state.

    ``evaluation_tools`` exists only for frozen surface experiments. Production
    constructors leave it empty so the catalog is discovered through the fixed
    MCP tools instead of being eagerly expanded into hundreds of definitions.
    """

    @asynccontextmanager
    async def lifespan(_server: MCPServer[AppState]) -> AsyncIterator[AppState]:
        delivery = state.result_delivery
        if delivery is None:
            yield state
            return
        with delivery.session():
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(delivery.expire_periodically)
                try:
                    yield state
                finally:
                    tasks.cancel_scope.cancel()

    server: MCPServer[AppState] = MCPServer(
        name="jacobian",
        title="Jacobian Mathematical Workbench",
        description=SERVER_DESCRIPTION,
        instructions=SERVER_INSTRUCTIONS,
        version=__version__,
        lifespan=lifespan,
        token_verifier=token_verifier,
        auth=auth,
        tools=list(evaluation_tools),
    )
    if state.result_delivery is not None:
        server.middleware.insert(0, state.result_delivery)
    register_core_projection(server, state)
    return server


__all__ = ["create_server"]
