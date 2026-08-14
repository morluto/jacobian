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

from mcp.server import MCPServer
from mcp.server.auth.provider import AccessToken

from jacobian import __version__
from jacobian.adapters.mcp.context import (
    AppState,
    AuthenticationError,
    RuntimeAccess,
    TenantRuntimeLimitError,
    _configured_root,
)
from jacobian.adapters.mcp.deployment_identity import load_deployment_identity
from jacobian.adapters.mcp.server import _build_server
from jacobian.operation_visibility import OperationVisibilityPolicy
from jacobian.registry import CheckerRegistry
from jacobian.runtime.execution import (
    create_inline_serving_runtime,
    create_serving_runtime,
)
from jacobian.runtime.model import JacobianRuntime
from jacobian.serving_catalog import ServingCatalog
from jacobian.storage.repository import ArtifactRepository

_TENANT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
DEFAULT_MAX_TENANT_RUNTIMES = 32
DEFAULT_TENANT_IDLE_TIMEOUT_SECONDS = 900.0


class TenantRuntimeRouterClosedError(RuntimeError):
    """The tenant runtime owner is shutting down or closed."""


@dataclass(slots=True)
class _TenantRuntimeEntry:
    runtime: JacobianRuntime
    active_requests: int
    last_used: float


class TenantRuntimeHold:
    """Private host guard preventing eviction during an active request."""

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
    TenantRuntimeHold | tuple[str, _TenantRuntimeEntry] | Literal["CREATE", "WAIT"]
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
        allow_anonymous: bool = False,
        anonymous_tenant_id: str = "anonymous",
        operation_policy: OperationVisibilityPolicy | None = None,
        max_tenant_runtimes: int = DEFAULT_MAX_TENANT_RUNTIMES,
        idle_timeout_seconds: float = DEFAULT_TENANT_IDLE_TIMEOUT_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        runtime_factory: Callable[..., JacobianRuntime],
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
        self.allow_anonymous = allow_anonymous
        self.anonymous_tenant_id = anonymous_tenant_id
        self.operation_policy = operation_policy
        self.max_tenant_runtimes = max_tenant_runtimes
        self.idle_timeout_seconds = idle_timeout_seconds
        self._clock = clock
        self._runtime_factory = runtime_factory
        self._runtimes: dict[str, _TenantRuntimeEntry] = {}
        self._quarantined: dict[str, _TenantRuntimeEntry] = {}
        self._creating: set[str] = set()
        self._evicting: set[str] = set()
        self._evictions_in_flight = 0
        self._condition = threading.Condition()
        self._closing = False
        self._closed = False
        self._shutdown_in_flight = False

    def runtime_for(self, subject: str | None) -> JacobianRuntime:
        """Return a compatible runtime for non-request local callers."""

        hold = self.hold_for(subject)
        try:
            return hold.runtime
        finally:
            hold.release()

    def hold_for(self, subject: str | None) -> TenantRuntimeHold:
        """Acquire one tenant runtime, creating or evicting outside the lock."""

        tenant_key = self._tenant_key(subject)
        while True:
            with self._condition:
                if self._closing or self._closed:
                    raise TenantRuntimeRouterClosedError(
                        "tenant runtime router is closing"
                    )
                plan = self._plan_acquisition(tenant_key)
                if isinstance(plan, TenantRuntimeHold):
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
                operation_policy=self.operation_policy,
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
                active_requests=1,
                last_used=self._clock(),
            )
            self._runtimes[tenant_key] = entry
            self._condition.notify_all()
        return TenantRuntimeHold(self, tenant_key, runtime)

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
        if tenant_key in self._evicting:
            return "WAIT"
        entry = self._runtimes.get(tenant_key)
        if entry is not None and not (
            entry.active_requests == 0
            and now - entry.last_used >= self.idle_timeout_seconds
        ):
            entry.active_requests += 1
            entry.last_used = now
            return TenantRuntimeHold(self, tenant_key, entry.runtime)
        if entry is not None:
            return self._begin_eviction(tenant_key, self._runtimes.pop(tenant_key))
        quarantined = self._quarantined.pop(tenant_key, None)
        if quarantined is not None:
            return self._begin_eviction(tenant_key, quarantined)
        if tenant_key in self._creating:
            return "WAIT"
        if (
            len(self._runtimes)
            + len(self._quarantined)
            + len(self._creating)
            + self._evictions_in_flight
            < self.max_tenant_runtimes
        ):
            self._creating.add(tenant_key)
            return "CREATE"
        if self._quarantined:
            eviction = min(
                self._quarantined.items(),
                key=lambda item: (item[1].last_used, item[0]),
            )
            del self._quarantined[eviction[0]]
            return self._begin_eviction(*eviction)
        inactive = tuple(
            (key, candidate)
            for key, candidate in self._runtimes.items()
            if candidate.active_requests == 0
        )
        if not inactive:
            raise TenantRuntimeLimitError(
                "This server has reached its in-memory tenant limit."
            )
        eviction = min(inactive, key=lambda item: (item[1].last_used, item[0]))
        del self._runtimes[eviction[0]]
        return self._begin_eviction(*eviction)

    def _begin_eviction(
        self, tenant_key: str, entry: _TenantRuntimeEntry
    ) -> tuple[str, _TenantRuntimeEntry]:
        self._evicting.add(tenant_key)
        self._evictions_in_flight += 1
        return tenant_key, entry

    def _close_evicted(self, eviction: tuple[str, _TenantRuntimeEntry]) -> None:
        tenant_key, entry = eviction
        try:
            entry.runtime.close()
        except BaseException:
            with self._condition:
                self._evictions_in_flight -= 1
                self._evicting.remove(tenant_key)
                self._quarantined[tenant_key] = entry
                self._condition.notify_all()
            raise
        with self._condition:
            self._evictions_in_flight -= 1
            self._evicting.remove(tenant_key)
            self._condition.notify_all()

    def _release(self, tenant_key: str) -> None:
        with self._condition:
            entry = self._runtimes.get(tenant_key)
            if entry is None or entry.active_requests < 1:
                raise RuntimeError("tenant runtime request ownership was lost")
            entry.active_requests -= 1
            entry.last_used = self._clock()
            self._condition.notify_all()

    def _collect_shutdown_runtimes(
        self,
    ) -> tuple[tuple[str, _TenantRuntimeEntry], ...]:
        with self._condition:
            while (
                self._creating
                or self._evictions_in_flight
                or any(entry.active_requests for entry in self._runtimes.values())
            ):
                self._condition.wait()
            return tuple(self._runtimes.items()) + tuple(self._quarantined.items())

    def _close_runtime_entries(
        self, runtimes: tuple[tuple[str, _TenantRuntimeEntry], ...]
    ) -> None:
        failures: list[BaseException] = []
        for tenant_key, entry in runtimes:
            try:
                entry.runtime.close()
            except BaseException as exc:
                failures.append(exc)
            else:
                with self._condition:
                    self._runtimes.pop(tenant_key, None)
                    self._quarantined.pop(tenant_key, None)
        if failures:
            exception_failures = [
                failure for failure in failures if isinstance(failure, Exception)
            ]
            if len(exception_failures) == len(failures):
                raise ExceptionGroup(
                    "one or more tenant runtimes failed to close", exception_failures
                )
            raise BaseExceptionGroup(
                "one or more tenant runtimes failed to close", failures
            )

    def _finish_shutdown(self) -> None:
        with self._condition:
            self._shutdown_in_flight = False
            self._closed = True
            self._closing = False
            self._condition.notify_all()

    def _abort_shutdown(self) -> None:
        with self._condition:
            self._shutdown_in_flight = False
            self._condition.notify_all()

    def close(self) -> None:
        """Close every tenant runtime owned by this router."""

        owns_shutdown = False
        try:
            with self._condition:
                if self._closed:
                    return
                while self._shutdown_in_flight:
                    self._condition.wait()
                    if self._closed:
                        return
                self._closing = True
                owns_shutdown = True
                self._shutdown_in_flight = True
            self._close_runtime_entries(self._collect_shutdown_runtimes())
            self._finish_shutdown()
        except BaseException:
            if owns_shutdown:
                self._abort_shutdown()
            raise


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
    state_dir: str | Path | None = None,
    *,
    allow_anonymous: bool = False,
    anonymous_tenant_id: str = "anonymous",
    token_verifier: Any | None = None,
    auth: Any | None = None,
    operation_policy: OperationVisibilityPolicy | None = None,
    max_tenant_runtimes: int | None = None,
    tenant_idle_timeout_seconds: float | None = None,
    runtime_factory: Callable[..., JacobianRuntime] | None = None,
) -> MCPServer[AppState]:
    """Create one remote host routing requests to isolated tenant runtimes."""

    from mcp.server.auth.middleware.auth_context import get_access_token

    root = _configured_root(state_dir)
    policy = operation_policy or OperationVisibilityPolicy()
    catalog = ServingCatalog.open(
        root / "metadata.sqlite3",
        policy,
        expected_package_version=__version__,
    )
    shared_store: ArtifactRepository | None = None
    selected_factory = runtime_factory
    if selected_factory is None and catalog.overlay is None:

        def selected_factory(
            tenant_root: str | Path,
            **_options: object,
        ) -> JacobianRuntime:
            del tenant_root, _options
            return create_inline_serving_runtime(catalog)

    elif selected_factory is None:
        shared_store = ArtifactRepository(root)
        shared_checkers = CheckerRegistry(shared_store)

        def selected_factory(
            tenant_root: str | Path,
            **_options: object,
        ) -> JacobianRuntime:
            return _create_tenant_runtime(
                Path(tenant_root),
                catalog,
                shared_checkers,
                policy,
            )

    router = TenantRuntimeRouter(
        root,
        allow_anonymous=allow_anonymous,
        anonymous_tenant_id=anonymous_tenant_id,
        operation_policy=policy,
        max_tenant_runtimes=(
            DEFAULT_MAX_TENANT_RUNTIMES
            if max_tenant_runtimes is None
            else max_tenant_runtimes
        ),
        idle_timeout_seconds=(
            DEFAULT_TENANT_IDLE_TIMEOUT_SECONDS
            if tenant_idle_timeout_seconds is None
            else tenant_idle_timeout_seconds
        ),
        runtime_factory=selected_factory,
    )

    def acquire_runtime(_operation_id: str | None = None) -> RuntimeAccess:
        access_token = get_access_token()
        subject = access_token.subject if access_token is not None else None
        hold = router.hold_for(subject)
        return RuntimeAccess(hold.runtime, hold.release)

    state = AppState(
        acquire_runtime=acquire_runtime,
        operation_catalog=catalog,
    )

    def close_owner() -> None:
        try:
            router.close()
        finally:
            if shared_store is not None:
                shared_store.close()

    return _build_server(
        state=state,
        close_owner=close_owner,
        deployment_identity=load_deployment_identity(),
        token_verifier=token_verifier,
        auth=auth,
    )


def _create_tenant_runtime(
    tenant_root: Path,
    catalog: ServingCatalog,
    shared_checkers: CheckerRegistry,
    operation_policy: OperationVisibilityPolicy,
) -> JacobianRuntime:
    """Create tenant-owned artifacts over deployment-owned mathematical state."""

    return create_serving_runtime(
        tenant_root,
        catalog,
        operation_policy=operation_policy,
        checker_registry=shared_checkers,
    )
