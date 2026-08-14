"""MCP request context: AppState and catalog/index access helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mcp.server.mcpserver import Context

from jacobian.serving_catalog import ServingCatalog


@dataclass(frozen=True, slots=True)
class AppState:
    operation_catalog: ServingCatalog
    authorize: Any = None


class AuthenticationError(PermissionError):
    """A remote request lacks a usable authenticated tenant subject."""


def _state(ctx: Context[Any, Any]) -> AppState:
    """Return the AppState for the current request."""

    state = ctx.request_context.lifespan_context
    if not isinstance(state, AppState):
        raise RuntimeError(
            "Jacobian is unavailable for this request. Retry once; if it fails "
            "again, inspect the local Jacobian log."
        )
    return state


def _catalog(ctx: Context[Any, Any]) -> ServingCatalog:
    """Return the serving catalog for discovery and inspection."""

    return _state(ctx).operation_catalog


def _authorize(ctx: Context[Any, Any]) -> None:
    """Run request-scoped authorization if the host provided one."""

    callback = _state(ctx).authorize
    if callback is not None:
        callback()


if __name__ == "__main__":
    from jacobian.adapters.mcp.cli import main

    main()
