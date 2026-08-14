"""Local stateless MCP host."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from mcp.server import MCPServer

from jacobian import __version__
from jacobian.adapters.mcp.context import AppState
from jacobian.adapters.mcp.core import register_core_projection
from jacobian.adapters.mcp.guidance import SERVER_DESCRIPTION, SERVER_INSTRUCTIONS
from jacobian.adapters.mcp.lifecycle import (
    runtime_lifespan,
)
from jacobian.serving_catalog import ServingCatalog


def create_server() -> MCPServer[AppState]:
    """Create the stateless local host over the immutable operation library.

    Built-in math neither reads nor creates server-owned state.
    """

    catalog = ServingCatalog.open()
    state = AppState(
        operation_catalog=catalog,
    )
    return _build_server(
        state=state,
        close_owner=_noop,
    )


def create_server_from_state(
    state: AppState,
    *,
    close_owner: Callable[[], None],
    start_owner: Callable[[], None] | None = None,
    token_verifier: Any | None = None,
    auth: Any | None = None,
) -> MCPServer[AppState]:
    """Register the local MCP projection over an explicitly owned state."""

    return _build_server(
        state=state,
        close_owner=close_owner,
        start_owner=start_owner,
        token_verifier=token_verifier,
        auth=auth,
    )


def _build_server(
    *,
    state: AppState,
    close_owner: Callable[[], None],
    start_owner: Callable[[], None] | None = None,
    token_verifier: Any | None = None,
    auth: Any | None = None,
) -> MCPServer[AppState]:
    """Register Jacobian's fixed MCP projection over one state owner."""

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
    register_core_projection(server, state)
    return server


def _noop() -> None:
    pass


if __name__ == "__main__":
    from jacobian.adapters.mcp.cli import main

    main()


__all__ = ["create_server", "create_server_from_state"]
