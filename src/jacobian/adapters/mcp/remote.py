"""Authentication and tenant routing for remote MCP transports."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from mcp.server.auth.provider import AccessToken

from jacobian.capabilities import CapabilityPolicy
from jacobian.runtime import CheckerAuthorityMode, create_runtime
from jacobian.runtime.model import JacobianRuntime

_TENANT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
DEFAULT_MAX_TENANT_RUNTIMES = 32
DEFAULT_TENANT_IDLE_TIMEOUT_SECONDS = 900.0


class AuthenticationError(PermissionError):
    """A remote request lacks a usable authenticated tenant subject."""


class TenantRuntimeLimitError(RuntimeError):
    """The server cannot admit another in-memory tenant runtime."""


class TenantRuntimeRouterClosedError(RuntimeError):
    """The tenant runtime owner is shutting down or closed."""


@dataclass(slots=True)
class _TenantRuntimeEntry:
    runtime: JacobianRuntime
    active_leases: int
    last_used: float


class TenantRuntimeLease:
    """One caller-owned hold preventing eviction or shutdown of a runtime."""

    def __init__(
        self,
        router: TenantRuntimeRouter,
        tenant_key: str,
        runtime: JacobianRuntime,
    ) -> None:
        self._router = router
        self._tenant_key = tenant_key
        self.runtime = runtime
        self._released = False

    def __enter__(self) -> JacobianRuntime:
        return self.runtime

    def __exit__(self, *_exc: object) -> None:
        self.release()

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._router._release(self._tenant_key)


type _AcquisitionPlan = (
    TenantRuntimeLease | tuple[str, _TenantRuntimeEntry] | Literal["CREATE", "WAIT"]
)


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


class TenantRuntimeRouter:
    """Own one isolated runtime per authenticated subject."""

    def __init__(
        self,
        root: str | Path,
        *,
        checker_authority: CheckerAuthorityMode = CheckerAuthorityMode.INSTALL_BUNDLED,
        allow_anonymous: bool = False,
        anonymous_tenant_id: str = "anonymous",
        capability_adapter_entrypoints: tuple[str, ...] = (),
        capability_policy: CapabilityPolicy | None = None,
        max_tenant_runtimes: int = DEFAULT_MAX_TENANT_RUNTIMES,
        idle_timeout_seconds: float = DEFAULT_TENANT_IDLE_TIMEOUT_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        runtime_factory: Callable[..., JacobianRuntime] = create_runtime,
    ) -> None:
        if max_tenant_runtimes < 1:
            raise ValueError("max_tenant_runtimes must be positive")
        if idle_timeout_seconds <= 0:
            raise ValueError("idle_timeout_seconds must be positive")
        if not _TENANT_PATTERN.fullmatch(anonymous_tenant_id):
            raise ValueError(
                "anonymous_tenant_id must start with a letter or digit, contain only "
                "letters, digits, '.', '_', or '-', and be at most 128 characters"
            )
        self.root = Path(root)
        self.checker_authority = checker_authority
        self.allow_anonymous = allow_anonymous
        self.anonymous_tenant_id = anonymous_tenant_id
        self.capability_adapter_entrypoints = capability_adapter_entrypoints
        self.capability_policy = capability_policy
        self.max_tenant_runtimes = max_tenant_runtimes
        self.idle_timeout_seconds = idle_timeout_seconds
        self._clock = clock
        self._runtime_factory = runtime_factory
        self._runtimes: dict[str, _TenantRuntimeEntry] = {}
        self._creating: set[str] = set()
        self._evictions_in_flight = 0
        self._condition = threading.Condition()
        self._closing = False
        self._closed = False
        self._shutdown_in_flight = False

    def runtime_for(self, subject: str | None) -> JacobianRuntime:
        """Return a compatible unleased runtime for non-request local callers."""

        lease = self.lease_for(subject)
        try:
            return lease.runtime
        finally:
            lease.release()

    def lease_for(self, subject: str | None) -> TenantRuntimeLease:
        """Acquire one tenant runtime, creating or evicting outside the lock."""

        tenant_key = self._tenant_key(subject)
        while True:
            with self._condition:
                if self._closing or self._closed:
                    raise TenantRuntimeRouterClosedError(
                        "tenant runtime router is closing"
                    )
                plan = self._plan_acquisition(tenant_key)
                if isinstance(plan, TenantRuntimeLease):
                    return plan
                if plan == "WAIT":
                    self._condition.wait()
                    continue
                if plan == "CREATE":
                    break
            self._close_evicted(plan)

        try:
            runtime = self._runtime_factory(
                self.root / "tenants" / tenant_key,
                checker_authority=self.checker_authority,
                capability_adapter_entrypoints=self.capability_adapter_entrypoints,
                capability_policy=self.capability_policy,
            )
        except BaseException:
            with self._condition:
                self._creating.discard(tenant_key)
                self._condition.notify_all()
            raise
        with self._condition:
            self._creating.discard(tenant_key)
            entry = _TenantRuntimeEntry(
                runtime=runtime,
                active_leases=1,
                last_used=self._clock(),
            )
            self._runtimes[tenant_key] = entry
            self._condition.notify_all()
        return TenantRuntimeLease(self, tenant_key, runtime)

    def _tenant_key(self, subject: str | None) -> str:
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
        return hashlib.sha256(tenant.encode("utf-8")).hexdigest()

    def _plan_acquisition(self, tenant_key: str) -> _AcquisitionPlan:
        now = self._clock()
        entry = self._runtimes.get(tenant_key)
        if entry is not None and not (
            entry.active_leases == 0
            and now - entry.last_used >= self.idle_timeout_seconds
        ):
            entry.active_leases += 1
            entry.last_used = now
            return TenantRuntimeLease(self, tenant_key, entry.runtime)
        if entry is not None:
            eviction = (tenant_key, self._runtimes.pop(tenant_key))
            self._evictions_in_flight += 1
            return eviction
        if tenant_key in self._creating:
            return "WAIT"
        if (
            len(self._runtimes) + len(self._creating) + self._evictions_in_flight
            < self.max_tenant_runtimes
        ):
            self._creating.add(tenant_key)
            return "CREATE"
        inactive = tuple(
            (key, candidate)
            for key, candidate in self._runtimes.items()
            if candidate.active_leases == 0
        )
        if not inactive:
            raise TenantRuntimeLimitError(
                "This server has reached its in-memory tenant limit."
            )
        eviction = min(inactive, key=lambda item: (item[1].last_used, item[0]))
        del self._runtimes[eviction[0]]
        self._evictions_in_flight += 1
        return eviction

    def _close_evicted(self, eviction: tuple[str, _TenantRuntimeEntry]) -> None:
        tenant_key, entry = eviction
        try:
            entry.runtime.close()
        except BaseException:
            with self._condition:
                self._evictions_in_flight -= 1
                self._runtimes[tenant_key] = entry
                self._condition.notify_all()
            raise
        with self._condition:
            self._evictions_in_flight -= 1
            self._condition.notify_all()

    def _release(self, tenant_key: str) -> None:
        with self._condition:
            entry = self._runtimes.get(tenant_key)
            if entry is None or entry.active_leases < 1:
                raise RuntimeError("tenant runtime lease ownership was lost")
            entry.active_leases -= 1
            entry.last_used = self._clock()
            self._condition.notify_all()

    def close(self) -> None:
        """Close every tenant runtime owned by this router."""

        with self._condition:
            if self._closed:
                return
            while self._shutdown_in_flight:
                self._condition.wait()
                if self._closed:
                    return
            self._closing = True
            while (
                self._creating
                or self._evictions_in_flight
                or any(entry.active_leases for entry in self._runtimes.values())
            ):
                self._condition.wait()
            self._shutdown_in_flight = True
            runtimes = tuple(self._runtimes.items())
        failures: list[Exception] = []
        for tenant_key, entry in runtimes:
            try:
                entry.runtime.close()
            except Exception as exc:
                failures.append(exc)
            else:
                with self._condition:
                    self._runtimes.pop(tenant_key, None)
        if failures:
            with self._condition:
                self._shutdown_in_flight = False
                self._condition.notify_all()
            raise ExceptionGroup(
                "one or more tenant runtimes failed to close", failures
            )
        with self._condition:
            self._shutdown_in_flight = False
            self._closed = True
            self._closing = False
            self._condition.notify_all()


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
