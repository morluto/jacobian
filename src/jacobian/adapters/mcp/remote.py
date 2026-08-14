"""Authentication and shared runtime ownership for remote MCP transports."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp.server import MCPServer
from mcp.server.auth.provider import AccessToken

from jacobian.adapters.mcp.context import (
    AppState,
    AuthenticationError,
    RuntimeAccess,
)
from jacobian.adapters.mcp.deployment_identity import load_deployment_identity
from jacobian.adapters.mcp.server import _build_server
from jacobian.operation_visibility import OperationVisibilityPolicy
from jacobian.runtime.execution import create_inline_serving_runtime
from jacobian.runtime.model import JacobianRuntime
from jacobian.serving_catalog import ServingCatalog

_TENANT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class _SharedRuntimeOwner:
    """Compile-safe seam owning one shared immutable runtime for all requests.

    Authentication and request-scoped tenant identity/scopes are resolved on
    every ``acquire`` from the MCP access token before any runtime is
    constructed. The runtime is built lazily on the first authenticated
    request and then shared immutably; there is no per-tenant state rooting,
    LRU, idle eviction, quarantine, or coordinated shutdown. ``release`` is a
    no-op because the runtime is shared and immutable for the host lifetime.
    """

    def __init__(
        self,
        runtime_factory: Callable[[], JacobianRuntime],
        *,
        allow_anonymous: bool = False,
        anonymous_tenant_id: str = "anonymous",
    ) -> None:
        if not _TENANT_PATTERN.fullmatch(anonymous_tenant_id):
            raise ValueError(
                "anonymous_tenant_id must start with a letter or digit, contain only "
                "letters, digits, '.', '_', or '-', and be at most 128 characters"
            )
        self._runtime_factory = runtime_factory
        self._allow_anonymous = allow_anonymous
        self._anonymous_tenant_id = anonymous_tenant_id
        self._runtime: JacobianRuntime | None = None
        self._lock = threading.Lock()

    def acquire(self, subject: str | None) -> RuntimeAccess:
        """Resolve request-scoped tenant identity, then return the shared runtime."""

        self._resolve_tenant(subject)
        return RuntimeAccess(self._ensure_runtime())

    def _resolve_tenant(self, subject: str | None) -> str:
        tenant = subject
        if tenant is None:
            if not self._allow_anonymous:
                raise AuthenticationError(
                    "Authentication is required for this server. "
                    "Authenticate with a configured bearer token and retry."
                )
            tenant = self._anonymous_tenant_id
        if not _TENANT_PATTERN.fullmatch(tenant):
            raise AuthenticationError(
                "The authenticated subject cannot be used for tenant isolation. "
                "Check the server token configuration, then authenticate again."
            )
        return tenant

    def _ensure_runtime(self) -> JacobianRuntime:
        with self._lock:
            if self._runtime is None:
                self._runtime = self._runtime_factory()
            return self._runtime

    def close(self) -> None:
        """Close the shared runtime, if one was constructed."""

        with self._lock:
            runtime = self._runtime
            self._runtime = None
        if runtime is not None:
            runtime.close()


@dataclass(frozen=True, slots=True)
class StaticTokenGrant:
    tenant_id: str
    token: str
    scopes: tuple[str, ...] = ("jacobian:use",)

    def __post_init__(self) -> None:
        if not _TENANT_PATTERN.fullmatch(self.tenant_id):
            raise ValueError(
                "tenant_id must start with a letter or digit, contain only letters, "
                "digits, '.', '_', or '-', and be at most 128 characters"
            )
        if len(self.token) < 32:
            raise ValueError("remote bearer tokens must contain at least 32 characters")
        if not self.scopes:
            raise ValueError("a remote token grant requires at least one scope")


class StaticTokenVerifier:
    """Verify operator-provisioned opaque bearer tokens without logging them."""

    def __init__(self, grants: tuple[StaticTokenGrant, ...]) -> None:
        if not grants:
            raise ValueError("at least one token grant is required")
        if len({grant.tenant_id for grant in grants}) != len(grants):
            raise ValueError("tenant IDs in the token file must be unique")
        if len({grant.token for grant in grants}) != len(grants):
            raise ValueError("bearer tokens in the token file must be unique")
        self._grants = tuple(
            (hashlib.sha256(grant.token.encode("utf-8")).digest(), grant)
            for grant in grants
        )

    async def verify_token(self, token: str) -> AccessToken | None:
        token_digest = hashlib.sha256(token.encode("utf-8")).digest()
        for grant_digest, grant in self._grants:
            if hmac.compare_digest(token_digest, grant_digest):
                return AccessToken(
                    token=token,
                    client_id=f"jacobian-tenant:{grant.tenant_id}",
                    scopes=list(grant.scopes),
                    subject=grant.tenant_id,
                )
        return None


def load_static_token_file(path: str | Path) -> tuple[StaticTokenGrant, ...]:
    """Load a strict JSON token file intended to be mounted as a secret."""

    selected = Path(path)
    try:
        payload: Any = json.loads(selected.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Jacobian could not read the remote token file. Check that the file "
            "exists, is readable, and contains valid JSON, then retry."
        ) from exc
    if not isinstance(payload, dict) or set(payload) != {"tokens"}:
        raise ValueError("token file must contain only a tokens array")
    records = payload["tokens"]
    if not isinstance(records, list):
        raise ValueError("token file tokens must be an array")
    grants: list[StaticTokenGrant] = []
    for index, record in enumerate(records, start=1):
        grants.append(_parse_token_record(index, record))
    return tuple(grants)


def _parse_token_record(index: int, record: Any) -> StaticTokenGrant:
    if not isinstance(record, dict):
        raise ValueError(f"token grant {index} must be a JSON object")
    extra_fields = set(record) - {
        "tenant_id",
        "token",
        "scopes",
    }
    if extra_fields:
        fields = ", ".join(repr(field) for field in sorted(extra_fields))
        raise ValueError(
            f"unsupported field {fields} in token grant {index}; "
            "use only tenant_id, token, and scopes"
        )
    tenant_id = record.get("tenant_id")
    token = record.get("token")
    scopes = record.get("scopes", ["jacobian:use"])
    if not isinstance(tenant_id, str):
        raise ValueError(f"tenant_id in token grant {index} must be a string")
    if not isinstance(token, str):
        raise ValueError(f"token in token grant {index} must be a string")
    if not isinstance(scopes, list) or not all(
        isinstance(scope, str) and scope for scope in scopes
    ):
        raise ValueError(
            f"scopes in token grant {index} must be an array of non-empty strings"
        )
    try:
        return StaticTokenGrant(
            tenant_id=tenant_id,
            token=token,
            scopes=tuple(scopes),
        )
    except ValueError as exc:
        raise ValueError(f"token grant {index}: {exc}") from exc


def create_remote_server(
    *,
    allow_anonymous: bool = False,
    anonymous_tenant_id: str = "anonymous",
    token_verifier: Any | None = None,
    auth: Any | None = None,
    operation_policy: OperationVisibilityPolicy | None = None,
    runtime_factory: Callable[[], JacobianRuntime] | None = None,
) -> MCPServer[AppState]:
    """Create one stateless remote host serving a shared operation runtime."""

    from mcp.server.auth.middleware.auth_context import get_access_token

    policy = operation_policy or OperationVisibilityPolicy()
    catalog = ServingCatalog.open(policy=policy)

    if runtime_factory is not None:

        def build_runtime() -> JacobianRuntime:
            return runtime_factory()

    else:

        def build_runtime() -> JacobianRuntime:
            return create_inline_serving_runtime(catalog)

    owner = _SharedRuntimeOwner(
        build_runtime,
        allow_anonymous=allow_anonymous,
        anonymous_tenant_id=anonymous_tenant_id,
    )

    def acquire_runtime(_operation_id: str | None = None) -> RuntimeAccess:
        access_token = get_access_token()
        subject = access_token.subject if access_token is not None else None
        return owner.acquire(subject)

    state = AppState(
        acquire_runtime=acquire_runtime,
        operation_catalog=catalog,
    )

    return _build_server(
        state=state,
        close_owner=owner.close,
        deployment_identity=load_deployment_identity(),
        token_verifier=token_verifier,
        auth=auth,
    )
