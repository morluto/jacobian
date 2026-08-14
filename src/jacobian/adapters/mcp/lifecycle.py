"""Shared MCP runtime lifecycle for local and remote runtime owners."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from jacobian.adapters.mcp.context import AppState


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
        close_owner()


__all__ = ["runtime_lifespan"]
