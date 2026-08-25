"""Request-scoped state for local and remote MCP hosts."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol, cast

from mcp.server.mcpserver import Context

from jacobian.catalog.catalog import Catalog


@dataclass(frozen=True, slots=True)
class AppState:
    operation_catalog: Catalog
    authorize: Callable[[], None] | None = None


class AuthenticationError(PermissionError):
    """A remote request lacks a usable authenticated tenant subject."""


class _RequestCancellation(Protocol):
    """The pinned SDK's request-owned cancellation signal."""

    def is_set(self) -> bool: ...

    def wait(self) -> Awaitable[None]: ...


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


def _request_cancellation(ctx: Context[Any, Any]) -> _RequestCancellation:
    """Return the cancellation signal for this exact MCP request.

    MCP 2.0 exposes the signal on its dispatch context, while the higher-level
    ``mcpserver.Context`` currently retains that context behind
    ``ServerSession._request_outbound``.  Keep the pinned-SDK compatibility
    seam here so an SDK change fails closed instead of silently disconnecting
    request cancellation from subprocess ownership.
    """

    session = ctx.request_context.session
    outbound = getattr(session, "_request_outbound", None)
    signal = getattr(outbound, "cancel_requested", None)
    if (
        signal is None
        or not callable(getattr(signal, "is_set", None))
        or not callable(getattr(signal, "wait", None))
    ):
        raise RuntimeError("MCP request cancellation signal is unavailable")
    return cast(_RequestCancellation, signal)


__all__ = ["AppState", "AuthenticationError"]
