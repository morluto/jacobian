"""Local stateless MCP host."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

from mcp.server import MCPServer
from mcp.server.mcpserver.tools import Tool

from jacobian import __version__
from jacobian.catalog.catalog import Catalog
from jacobian.mcp.guidance import SERVER_DESCRIPTION, SERVER_INSTRUCTIONS
from jacobian.mcp.protocol import register_core_projection
from jacobian.mcp.runtime import AppState


def create_server() -> MCPServer[AppState]:
    """Create the stateless local host over the immutable operation library.

    Built-in math neither reads nor creates server-owned state.
    """

    catalog = Catalog.open()
    state = AppState(
        operation_catalog=catalog,
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
        yield state

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
    register_core_projection(server, state)
    return server


__all__ = ["create_server"]
