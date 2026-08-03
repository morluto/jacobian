"""In-process remote MCP tenant-router lifecycle boundary tests."""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path

import pytest

from jacobian.adapters.mcp.remote import (
    TenantRuntimeLimitError,
    TenantRuntimeRouter,
    TenantRuntimeRouterClosedError,
)
from jacobian.runtime import CheckerAuthorityMode
from jacobian.storage.errors import ArtifactNotFoundError


def test_tenant_router_isolates_artifact_stores(tmp_path: Path) -> None:
    router = TenantRuntimeRouter(
        tmp_path,
        checker_authority=CheckerAuthorityMode.NONE,
        max_tenant_runtimes=2,
    )
    alpha = router.runtime_for("alpha")
    beta = router.runtime_for("beta")
    stored = alpha.core.store.register_descriptor(
        kind="semantics",
        name="alpha-only",
        version="1",
        definition={"value": 1},
    )

    assert alpha.core.store.root != beta.core.store.root
    assert router.runtime_for("alpha") is alpha
    with pytest.raises(ArtifactNotFoundError):
        beta.core.store.get(stored)


class _FakeRuntime:
    def __init__(self, identity: Path, *, fail_close: bool = False) -> None:
        self.identity = identity
        self.fail_close = fail_close
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        if self.fail_close:
            raise RuntimeError("injected close failure")


def test_tenant_router_single_flights_one_tenant_and_parallelizes_distinct_tenants(
    tmp_path: Path,
) -> None:
    creation_barrier = threading.Barrier(2)
    created: list[_FakeRuntime] = []
    created_lock = threading.Lock()

    def factory(path: Path, **_kwargs: object) -> _FakeRuntime:
        creation_barrier.wait(timeout=2)
        runtime = _FakeRuntime(path)
        with created_lock:
            created.append(runtime)
        return runtime

    router = TenantRuntimeRouter(
        tmp_path,
        max_tenant_runtimes=3,
        runtime_factory=factory,  # type: ignore[arg-type]
    )
    leases = []

    def acquire(subject: str) -> None:
        leases.append(router.lease_for(subject))

    workers = [
        threading.Thread(target=acquire, args=(subject,))
        for subject in ("alpha", "alpha", "beta")
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=2)
        assert not worker.is_alive()

    alpha_runtimes = [
        lease.runtime
        for lease in leases
        if lease.runtime.identity.name == hashlib.sha256(b"alpha").hexdigest()
    ]
    assert len(created) == 2
    assert len(alpha_runtimes) == 2
    assert alpha_runtimes[0] is alpha_runtimes[1]
    for lease in leases:
        lease.release()


def test_tenant_router_uses_idle_ttl_lru_and_never_evicts_an_active_lease(
    tmp_path: Path,
) -> None:
    now = [0.0]
    created: list[_FakeRuntime] = []

    def factory(path: Path, **_kwargs: object) -> _FakeRuntime:
        runtime = _FakeRuntime(path)
        created.append(runtime)
        return runtime

    router = TenantRuntimeRouter(
        tmp_path,
        max_tenant_runtimes=2,
        idle_timeout_seconds=10,
        clock=lambda: now[0],
        runtime_factory=factory,  # type: ignore[arg-type]
    )
    alpha = router.lease_for("alpha")
    now[0] = 1
    beta_runtime = router.runtime_for("beta")
    now[0] = 2
    alpha.release()
    now[0] = 3
    alpha_runtime = router.runtime_for("alpha")
    now[0] = 4
    gamma_runtime = router.runtime_for("gamma")

    assert beta_runtime.close_calls == 1
    assert alpha_runtime.close_calls == 0
    assert gamma_runtime.close_calls == 0

    active = router.lease_for("alpha")
    now[0] = 20
    refreshed_gamma = router.runtime_for("gamma")
    assert refreshed_gamma is not gamma_runtime
    assert gamma_runtime.close_calls == 1
    active.release()


def test_tenant_router_restores_failed_eviction_and_shutdown_waits_for_leases(
    tmp_path: Path,
) -> None:
    first = _FakeRuntime(tmp_path / "first", fail_close=True)
    created = [first]

    def factory(path: Path, **_kwargs: object) -> _FakeRuntime:
        if len(created) == 1:
            return first
        runtime = _FakeRuntime(path)
        created.append(runtime)
        return runtime

    router = TenantRuntimeRouter(
        tmp_path,
        max_tenant_runtimes=1,
        runtime_factory=factory,  # type: ignore[arg-type]
    )
    assert router.runtime_for("alpha") is first
    with pytest.raises(RuntimeError, match="injected close failure"):
        router.runtime_for("beta")
    assert router.runtime_for("alpha") is first

    lease = router.lease_for("alpha")
    with pytest.raises(TenantRuntimeLimitError, match="tenant limit"):
        router.lease_for("beta")
    first.fail_close = False
    close_entered = threading.Event()
    original_close = router.close

    def close_after_signaling() -> None:
        close_entered.set()
        original_close()

    router.close = close_after_signaling  # type: ignore[method-assign]
    closed = threading.Event()
    closer = threading.Thread(target=lambda: (router.close(), closed.set()))
    closer.start()
    assert close_entered.wait(timeout=2)
    assert not closed.is_set()
    with pytest.raises(TenantRuntimeRouterClosedError, match="closing"):
        router.lease_for("beta")
    lease.release()
    closer.join(timeout=2)
    assert closed.is_set()


def test_tenant_router_retries_only_failed_runtime_shutdowns(tmp_path: Path) -> None:
    created: list[_FakeRuntime] = []

    def factory(path: Path, **_kwargs: object) -> _FakeRuntime:
        runtime = _FakeRuntime(path, fail_close=not created)
        created.append(runtime)
        return runtime

    router = TenantRuntimeRouter(
        tmp_path,
        max_tenant_runtimes=2,
        runtime_factory=factory,  # type: ignore[arg-type]
    )
    router.runtime_for("alpha")
    router.runtime_for("beta")

    with pytest.raises(ExceptionGroup, match="tenant runtimes failed to close"):
        router.close()
    assert [runtime.close_calls for runtime in created] == [1, 1]

    created[0].fail_close = False
    router.close()
    router.close()

    assert [runtime.close_calls for runtime in created] == [2, 1]


def test_anonymous_tenant_namespace_is_fixed_by_the_operator(tmp_path: Path) -> None:
    first = TenantRuntimeRouter(
        tmp_path,
        checker_authority=CheckerAuthorityMode.NONE,
        allow_anonymous=True,
        anonymous_tenant_id="test-endpoint-a",
    )
    second = TenantRuntimeRouter(
        tmp_path,
        checker_authority=CheckerAuthorityMode.NONE,
        allow_anonymous=True,
        anonymous_tenant_id="test-endpoint-b",
    )

    first_runtime = first.runtime_for(None)
    second_runtime = second.runtime_for(None)

    assert first_runtime.core.store.root != second_runtime.core.store.root
    assert first.runtime_for(None) is first_runtime
    with pytest.raises(ValueError, match="anonymous_tenant_id must start"):
        TenantRuntimeRouter(
            tmp_path,
            checker_authority=CheckerAuthorityMode.NONE,
            allow_anonymous=True,
            anonymous_tenant_id="caller controlled",
        )
