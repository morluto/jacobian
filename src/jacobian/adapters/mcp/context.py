"""MCP request context: AppState and runtime resolution helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any

from mcp.server.mcpserver import Context

from jacobian.adapters.mcp.tooling import AgentRecoveryError
from jacobian.runtime.model import JacobianRuntime


@dataclass(frozen=True, slots=True)
class AppState:
    acquire_runtime: Callable[[str | None], RuntimeAccess]
    operation_catalog: Any


@dataclass(frozen=True, slots=True)
class RuntimeAccess:
    """A lazily acquired runtime and its private host release callback."""

    runtime: JacobianRuntime
    release: Callable[[], None] | None = None


class AuthenticationError(PermissionError):
    """A remote request lacks a usable authenticated tenant subject."""


_active_runtime: ContextVar[JacobianRuntime | None] = ContextVar(
    "jacobian_mcp_active_runtime",
    default=None,
)


@contextmanager
def _runtime(
    ctx: Context[Any, Any], operation_id: str | None = None
) -> Iterator[JacobianRuntime]:
    """Return the shared runtime for the current request."""

    active_runtime = _active_runtime.get()
    if active_runtime is not None:
        yield active_runtime
        return

    state = ctx.request_context.lifespan_context
    if not isinstance(state, AppState):
        raise AgentRecoveryError(
            "Jacobian is unavailable for this request. Retry once; if it fails "
            "again, inspect the local Jacobian log."
        )
    with _runtime_scope(state, operation_id) as runtime:
        yield runtime


def _catalog(ctx: Context[Any, Any]) -> Any:
    """Return the deployment catalog without acquiring an execution runtime."""

    state = ctx.request_context.lifespan_context
    if not isinstance(state, AppState):
        raise AgentRecoveryError(
            "Jacobian is unavailable for this request. Retry once; if it fails "
            "again, inspect the local Jacobian log."
        )
    return state.operation_catalog


@contextmanager
def _runtime_scope(
    state: AppState, operation_id: str | None = None
) -> Iterator[JacobianRuntime]:
    """Bind exactly one runtime and blocking-worker owner to an MCP request."""

    access = state.acquire_runtime(operation_id)
    token: Token[JacobianRuntime | None] = _active_runtime.set(access.runtime)
    try:
        yield access.runtime
    finally:
        _active_runtime.reset(token)
        if access.release is not None:
            access.release()


if __name__ == "__main__":
    from jacobian.adapters.mcp.cli import main

    main()
