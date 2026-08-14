"""Authentication and stateless serving for remote MCP transports."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp.server import MCPServer
from mcp.server.auth.provider import AccessToken

from jacobian.adapters.mcp.context import (
    AppState,
    AuthenticationError,
)
from jacobian.adapters.mcp.server import _build_server
from jacobian.serving_catalog import ServingCatalog

_TENANT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


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


def _resolve_tenant(
    subject: str | None,
    *,
    allow_anonymous: bool,
    anonymous_tenant_id: str,
) -> str:
    """Validate the authenticated subject for request-scoped tenant identity."""

    tenant = subject
    if tenant is None:
        if not allow_anonymous:
            raise AuthenticationError(
                "Authentication is required for this server. "
                "Authenticate with a configured bearer token and retry."
            )
        tenant = anonymous_tenant_id
    if not _TENANT_PATTERN.fullmatch(tenant):
        raise AuthenticationError(
            "The authenticated subject cannot be used for tenant isolation. "
            "Check the server token configuration, then authenticate again."
        )
    return tenant


def create_remote_server(
    *,
    allow_anonymous: bool = False,
    anonymous_tenant_id: str = "anonymous",
    token_verifier: Any | None = None,
    auth: Any | None = None,
) -> MCPServer[AppState]:
    """Create one stateless remote host serving the immutable operation library."""

    from mcp.server.auth.middleware.auth_context import get_access_token

    if not _TENANT_PATTERN.fullmatch(anonymous_tenant_id):
        raise ValueError(
            "anonymous_tenant_id must start with a letter or digit, contain only "
            "letters, digits, '.', '_', or '-', and be at most 128 characters"
        )

    catalog = ServingCatalog.open()

    def authorize() -> None:
        access_token = get_access_token()
        subject = access_token.subject if access_token is not None else None
        _resolve_tenant(
            subject,
            allow_anonymous=allow_anonymous,
            anonymous_tenant_id=anonymous_tenant_id,
        )

    state = AppState(
        operation_catalog=catalog,
        authorize=authorize,
    )

    return _build_server(
        state=state,
        close_owner=_noop,
        token_verifier=token_verifier,
        auth=auth,
    )


def _noop() -> None:
    pass
