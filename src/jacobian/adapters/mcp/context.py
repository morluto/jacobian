"""MCP request context: AppState and runtime resolution helpers."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError

from jacobian.adapters.mcp.remote import TenantRuntimeRouter
from jacobian.adapters.mcp.tooling import (
    AgentRecoveryError,
    MCPBlockingWorkerRegistry,
    blocking_worker_scope,
)
from jacobian.runtime.model import JacobianRuntime

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AppState:
    runtime: JacobianRuntime | None
    worker_registry: MCPBlockingWorkerRegistry
    tenant_router: TenantRuntimeRouter | None = None


_active_runtime: ContextVar[JacobianRuntime | None] = ContextVar(
    "jacobian_mcp_active_runtime",
    default=None,
)


@contextmanager
def _runtime(ctx: Context[Any, Any]) -> Iterator[JacobianRuntime]:
    """Return a runtime, holding a tenant lease for the full request lifetime.

    When tenant isolation is active, the lease is held until the context
    manager exits so the runtime cannot be evicted mid-request.
    """

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
    with _runtime_scope(state) as runtime:
        yield runtime


@contextmanager
def _runtime_scope(state: AppState) -> Iterator[JacobianRuntime]:
    """Bind exactly one runtime and blocking-worker owner to an MCP request."""

    if state.tenant_router is not None:
        from mcp.server.auth.middleware.auth_context import get_access_token

        access_token = get_access_token()
        subject = access_token.subject if access_token is not None else None
        lease = state.tenant_router.lease_for(subject)
        with blocking_worker_scope(
            state.worker_registry,
            lease_release=lease.release,
        ):
            token: Token[JacobianRuntime | None] = _active_runtime.set(lease.runtime)
            try:
                _start_lean_warmup(lease.runtime)
                yield lease.runtime
            finally:
                _active_runtime.reset(token)
        return
    if state.runtime is None:
        raise AgentRecoveryError(
            "Jacobian is unavailable for this request. Retry once; if it fails "
            "again, inspect the local Jacobian log."
        )
    with blocking_worker_scope(state.worker_registry):
        token = _active_runtime.set(state.runtime)
        try:
            yield state.runtime
        finally:
            _active_runtime.reset(token)


def _start_lean_warmup(runtime: JacobianRuntime) -> None:
    if (
        runtime.portfolio.lean is not None
        and os.environ.get("JACOBIAN_LEAN_WARMUP") == "1"
    ):
        runtime.portfolio.lean.start_mathlib_warmup()


@contextmanager
def _static_resource_runtime(
    runtime: JacobianRuntime | None,
    tenant_router: TenantRuntimeRouter | None,
) -> Iterator[JacobianRuntime]:
    """Route SDK static resources through the active authentication context.

    MCP 2.0.0 does not inject ``Context`` into static resources, but its HTTP
    authentication middleware still scopes the access token with a contextvar.
    Template resources use native ``Context`` injection and ``_runtime`` instead.
    """

    active_runtime = _active_runtime.get()
    if active_runtime is not None:
        yield active_runtime
        return

    if tenant_router is not None:
        from mcp.server.auth.middleware.auth_context import get_access_token

        access_token = get_access_token()
        subject = access_token.subject if access_token is not None else None
        with tenant_router.lease_for(subject) as active_runtime:
            _start_lean_warmup(active_runtime)
            yield active_runtime
        return
    if runtime is None:
        raise AgentRecoveryError(
            "Jacobian is unavailable for this resource request. Retry once; if it "
            "fails again, inspect the local Jacobian log."
        )
    yield runtime


def _configured_root(state_dir: str | Path | None) -> Path:
    if state_dir is not None:
        return Path(state_dir)
    return Path(os.environ.get("JACOBIAN_STATE_DIR", ".jacobian"))


def _unwrap_tool_error(exc: Exception) -> Exception:
    current = exc
    for _ in range(4):
        if not isinstance(current, ToolError) or not isinstance(
            current.__cause__, Exception
        ):
            return current
        current = current.__cause__
    return current


def _classify_public_tool_error(
    tool_name: str, tool_error: Exception
) -> tuple[str, str, str]:
    from jacobian.adapters.mcp.remote import (
        AuthenticationError,
        TenantRuntimeLimitError,
    )
    from jacobian.registry import CheckerNotFoundError
    from jacobian.storage.errors import ArtifactNotFoundError

    if isinstance(tool_error, AgentRecoveryError):
        return (
            "SERVICE_UNAVAILABLE",
            str(tool_error),
            "Follow the recovery action in the message, then retry the tool.",
        )
    if isinstance(tool_error, TimeoutError):
        return (
            "OPERATION_TIMED_OUT",
            "The operation did not finish within the allowed time.",
            "Retry with a larger time budget or a smaller request.",
        )
    if isinstance(tool_error, AuthenticationError):
        return (
            "AUTHENTICATION_REQUIRED",
            str(tool_error),
            "Authenticate with a configured bearer token, then retry.",
        )
    if isinstance(tool_error, TenantRuntimeLimitError):
        return (
            "TENANT_KERNEL_LIMIT",
            str(tool_error),
            "Retry on another server instance or ask the operator to raise the limit.",
        )
    if isinstance(tool_error, PermissionError):
        return (
            "PERMISSION_DENIED",
            "Jacobian could not access the required local resource.",
            "Check the state-directory permissions, then retry.",
        )
    if isinstance(
        tool_error,
        (
            ArtifactNotFoundError,
            CheckerNotFoundError,
        ),
    ):
        return (
            "RESOURCE_NOT_FOUND",
            "A required Jacobian resource was not found.",
            ("Check the artifact URI returned by the earlier tool call, then retry."),
        )
    if isinstance(tool_error, ValueError):
        return (
            "INVALID_INPUT",
            "The tool input is not valid for this operation.",
            "Check the tool input schema or call math.find, then retry.",
        )
    return (
        "OPERATION_FAILED",
        "Jacobian could not complete the operation.",
        "Retry once; if it fails again, inspect the local Jacobian log.",
    )


def _public_tool_error(tool_name: str, exc: Exception) -> str:
    code, message, hint = _classify_public_tool_error(
        tool_name, _unwrap_tool_error(exc)
    )
    return json.dumps(
        {
            "error": {
                "code": code,
                "stage": tool_name,
                "message": message,
                "hint": hint,
            }
        },
        ensure_ascii=False,
        sort_keys=True,
    )


if __name__ == "__main__":
    from jacobian.adapters.mcp.cli import main

    main()
