"""Shared MCP runtime lifecycle for local and remote runtime owners."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from jacobian.adapters.mcp.context import AppState
from jacobian.adapters.mcp.tooling import MCPBlockingWorkerShutdownError
from jacobian.runtime import CheckerAuthorityMode


def selected_checker_authority(
    authority: CheckerAuthorityMode | None,
) -> CheckerAuthorityMode:
    return authority or CheckerAuthorityMode.INSTALL_BUNDLED


@asynccontextmanager
async def runtime_lifespan(
    _server: Any,
    *,
    state: AppState,
    close_owner: Callable[[], None],
    start_owner: Callable[[], None] | None = None,
) -> AsyncIterator[AppState]:
    if start_owner is not None:
        start_owner()
    try:
        yield state
    finally:
        try:
            await state.worker_registry.close()
        except MCPBlockingWorkerShutdownError as exc:
            state.worker_registry.defer_until_quiescent(close_owner)
            raise exc from None
        close_owner()


__all__ = ["runtime_lifespan", "selected_checker_authority"]
