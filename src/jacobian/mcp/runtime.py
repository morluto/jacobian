"""Request-scoped state for local and remote MCP hosts."""

from __future__ import annotations

from _thread import LockType
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from mcp.server.mcpserver import Context

from jacobian.catalog.catalog import Catalog


@dataclass(frozen=True, slots=True)
class AppState:
    operation_catalog: Catalog
    authorize: Callable[[], None] | None = None
    execution_lock: LockType = field(default_factory=Lock, repr=False, compare=False)


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


def _catalog(ctx: Context[Any, Any]) -> Catalog:
    """Return the serving catalog for discovery and inspection."""

    return _state(ctx).operation_catalog


def _authorize(ctx: Context[Any, Any]) -> None:
    """Run request-scoped authorization if the host provided one."""

    callback = _state(ctx).authorize
    if callback is not None:
        callback()


__all__ = ["AppState", "AuthenticationError"]
