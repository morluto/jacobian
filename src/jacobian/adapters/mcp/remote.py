"""Authentication and tenant routing for remote MCP transports."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp.server.auth.provider import AccessToken

from jacobian.capabilities import CapabilityPolicy
from jacobian.kernel import JacobianKernel

_TENANT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
DEFAULT_MAX_TENANT_KERNELS = 32


class AuthenticationError(PermissionError):
    """A remote request lacks a usable authenticated tenant subject."""


class TenantKernelLimitError(RuntimeError):
    """The server cannot admit another in-memory tenant kernel."""


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
        self._grants = grants

    async def verify_token(self, token: str) -> AccessToken | None:
        for grant in self._grants:
            if hmac.compare_digest(token, grant.token):
                return AccessToken(
                    token=token,
                    client_id=f"jacobian-tenant:{grant.tenant_id}",
                    scopes=list(grant.scopes),
                    subject=grant.tenant_id,
                )
        return None


class TenantKernelRouter:
    """Create one isolated kernel root per authenticated subject."""

    def __init__(
        self,
        root: str | Path,
        *,
        install_references: bool = True,
        allow_anonymous: bool = False,
        anonymous_tenant_id: str = "anonymous",
        capability_adapter_entrypoints: tuple[str, ...] = (),
        capability_policy: CapabilityPolicy | None = None,
        max_tenant_kernels: int = DEFAULT_MAX_TENANT_KERNELS,
    ) -> None:
        if max_tenant_kernels < 1:
            raise ValueError("max_tenant_kernels must be positive")
        if not _TENANT_PATTERN.fullmatch(anonymous_tenant_id):
            raise ValueError(
                "anonymous_tenant_id must start with a letter or digit, contain only "
                "letters, digits, '.', '_', or '-', and be at most 128 characters"
            )
        self.root = Path(root)
        self.install_references = install_references
        self.allow_anonymous = allow_anonymous
        self.anonymous_tenant_id = anonymous_tenant_id
        self.capability_adapter_entrypoints = capability_adapter_entrypoints
        self.capability_policy = capability_policy
        self.max_tenant_kernels = max_tenant_kernels
        self._kernels: dict[str, JacobianKernel] = {}
        self._lock = threading.Lock()

    def kernel_for(self, subject: str | None) -> JacobianKernel:
        tenant = subject
        if tenant is None:
            if not self.allow_anonymous:
                raise AuthenticationError(
                    "Authentication is required for this server. "
                    "Authenticate with a configured bearer token and retry."
                )
            tenant = self.anonymous_tenant_id
        if not _TENANT_PATTERN.fullmatch(tenant):
            raise AuthenticationError(
                "The authenticated subject cannot be used for tenant isolation. "
                "Check the server token configuration, then authenticate again."
            )
        tenant_key = hashlib.sha256(tenant.encode("utf-8")).hexdigest()
        with self._lock:
            kernel = self._kernels.get(tenant_key)
            if kernel is None:
                if len(self._kernels) >= self.max_tenant_kernels:
                    raise TenantKernelLimitError(
                        "This server has reached its in-memory tenant limit."
                    )
                kernel = JacobianKernel(
                    self.root / "tenants" / tenant_key,
                    install_references=self.install_references,
                    capability_adapter_entrypoints=(
                        self.capability_adapter_entrypoints
                    ),
                    capability_policy=self.capability_policy,
                )
                self._kernels[tenant_key] = kernel
            return kernel


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
            grants.append(
                StaticTokenGrant(
                    tenant_id=tenant_id,
                    token=token,
                    scopes=tuple(scopes),
                )
            )
        except ValueError as exc:
            raise ValueError(f"token grant {index}: {exc}") from exc
    return tuple(grants)
