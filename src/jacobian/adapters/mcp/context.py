"""MCP request context: AppState and runtime resolution helpers."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError

from jacobian.adapters.mcp.remote import TenantRuntimeRouter
from jacobian.adapters.mcp.tooling import AgentRecoveryError
from jacobian.runtime.model import JacobianRuntime

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AppState:
    runtime: JacobianRuntime | None
    tenant_router: TenantRuntimeRouter | None = None


def _runtime(ctx: Context[AppState, Any] | None) -> JacobianRuntime:
    if ctx is None:
        raise AgentRecoveryError(
            "Jacobian is unavailable for this request. Retry once; if it fails "
            "again, inspect the local Jacobian log."
        )
    state = ctx.request_context.lifespan_context
    if not isinstance(state, AppState):
        raise AgentRecoveryError(
            "Jacobian is unavailable for this request. Retry once; if it fails "
            "again, inspect the local Jacobian log."
        )
    if state.tenant_router is not None:
        from mcp.server.auth.middleware.auth_context import get_access_token

        access_token = get_access_token()
        subject = access_token.subject if access_token is not None else None
        runtime = state.tenant_router.runtime_for(subject)
        _start_lean_warmup(runtime)
        return runtime
    if state.runtime is None:
        raise AgentRecoveryError(
            "Jacobian is unavailable for this request. Retry once; if it fails "
            "again, inspect the local Jacobian log."
        )
    return state.runtime


def _start_lean_warmup(runtime: JacobianRuntime) -> None:
    if (
        runtime.portfolio.lean is not None
        and os.environ.get("JACOBIAN_LEAN_WARMUP") == "1"
    ):
        runtime.portfolio.lean.start_mathlib_warmup()


@contextmanager
def _resource_runtime(
    runtime: JacobianRuntime | None,
    tenant_router: TenantRuntimeRouter | None,
) -> Iterator[JacobianRuntime]:
    """Route resources through the same auth context as tools.

    MCP 2.0.0 does not inject ``Context`` into static resources, but its HTTP
    authentication middleware still scopes the access token with a contextvar.
    """

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
    from jacobian.experiments import ExperimentNotFoundError
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
            ExperimentNotFoundError,
        ),
    ):
        return (
            "RESOURCE_NOT_FOUND",
            "A required Jacobian resource was not found.",
            (
                "Check the artifact or experiment URI returned by the earlier tool "
                "call, then retry."
            ),
        )
    if isinstance(tool_error, ValueError):
        return (
            "INVALID_INPUT",
            "The tool input is not valid for this operation.",
            "Check the tool input schema or call capability.describe, then retry.",
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


def _experiment_scope_content(runtime: JacobianRuntime, snapshot: Any) -> str:
    scope_uri = getattr(snapshot, "scope_uri", None)
    if scope_uri is None:
        return json.dumps(
            {
                "experiment_uri": snapshot.experiment_uri,
                "scope_uri": None,
            },
            sort_keys=True,
        )
    scope = runtime.core.store.get(scope_uri)
    return json.dumps(
        {
            "experiment_uri": snapshot.experiment_uri,
            "scope_uri": scope.artifact_uri,
            "manifest": scope.manifest.model_dump(mode="json"),
            "payload": scope.payload,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


if __name__ == "__main__":
    from jacobian.adapters.mcp.cli import main

    main()
