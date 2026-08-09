"""In-process remote MCP tenant-router lifecycle boundary tests."""

from __future__ import annotations

import hashlib
import threading
from contextlib import suppress
from pathlib import Path

import pytest

from jacobian.adapters.mcp.remote import (
    TenantRuntimeLease,
    TenantRuntimeRouter,
    TenantRuntimeRouterClosedError,
)
from jacobian.contracts.reasoning import ReasoningWriteRequest
from jacobian.reasoning_log import ReasoningProtocolError
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
    reasoning_run = alpha.core.reasoning_log.write(
        ReasoningWriteRequest(phase="PLAN", summary="Alpha-only plan.")
    )
    with pytest.raises(ReasoningProtocolError, match="does not exist"):
        beta.core.reasoning_log.inspect(reasoning_run.run_id)


class _FakeRuntime:
    def __init__(
        self,
        identity: Path,
        *,
        fail_close: bool = False,
        close_started: threading.Event | None = None,
        close_release: threading.Event | None = None,
    ) -> None:
        self.identity = identity
        self.fail_close = fail_close
        self.close_started = close_started
        self.close_release = close_release
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        should_fail = self.fail_close
        if self.close_started is not None:
            self.close_started.set()
        if self.close_release is not None:
            self.close_release.wait(timeout=2)
        if should_fail:
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


def test_tenant_router_quarantines_failed_eviction_until_shutdown_retry(
    tmp_path: Path,
) -> None:
    first = _FakeRuntime(tmp_path / "first", fail_close=True)
    created: list[_FakeRuntime] = []

    def factory(path: Path, **_kwargs: object) -> _FakeRuntime:
        runtime = first if not created else _FakeRuntime(path)
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

    assert first.close_calls == 1
    assert created == [first]
    with pytest.raises(RuntimeError, match="injected close failure"):
        router.runtime_for("gamma")
    assert first.close_calls == 2
    assert created == [first]

    first.fail_close = False
    router.close()

    assert first.close_calls == 3
    with pytest.raises(TenantRuntimeRouterClosedError, match="closing"):
        router.lease_for("beta")


def test_tenant_router_blocks_same_tenant_during_eviction_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    close_started = threading.Event()
    close_release = threading.Event()
    first = _FakeRuntime(
        tmp_path / "first",
        fail_close=True,
        close_started=close_started,
        close_release=close_release,
    )
    created: list[_FakeRuntime] = []

    def factory(path: Path, **_kwargs: object) -> _FakeRuntime:
        runtime = first if not created else _FakeRuntime(path)
        created.append(runtime)
        return runtime

    router = TenantRuntimeRouter(
        tmp_path,
        max_tenant_runtimes=1,
        runtime_factory=factory,  # type: ignore[arg-type]
    )
    router.runtime_for("alpha")

    eviction_error: list[BaseException] = []

    def evict() -> None:
        try:
            router.lease_for("beta")
        except BaseException as exc:
            eviction_error.append(exc)

    evictor = threading.Thread(target=evict)
    evictor.start()
    assert close_started.wait(timeout=2)

    acquired: list[TenantRuntimeLease] = []
    reacquirer_waiting = threading.Event()
    original_wait = router._condition.wait

    def observed_wait(timeout: float | None = None) -> bool:
        reacquirer_waiting.set()
        return original_wait(timeout)

    monkeypatch.setattr(router._condition, "wait", observed_wait)

    def reacquire() -> None:
        acquired.append(router.lease_for("alpha"))

    reacquirer = threading.Thread(target=reacquire)
    reacquirer.start()
    assert reacquirer_waiting.wait(timeout=2)
    assert acquired == []

    first.fail_close = False
    close_release.set()
    evictor.join(timeout=2)
    reacquirer.join(timeout=2)

    assert len(eviction_error) == 1
    assert len(acquired) == 1
    assert acquired[0].runtime is not first
    assert first.close_calls == 2
    assert created == [first, acquired[0].runtime]
    acquired[0].release()


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
    with pytest.raises(TenantRuntimeRouterClosedError, match="closing"):
        router.lease_for("gamma")

    created[0].fail_close = False
    router.close()
    router.close()

    assert [runtime.close_calls for runtime in created] == [2, 1]


def test_tenant_router_shutdown_is_retryable_after_base_exception(
    tmp_path: Path,
) -> None:
    class InterruptOnceRuntime(_FakeRuntime):
        def close(self) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                raise KeyboardInterrupt

    runtime = InterruptOnceRuntime(tmp_path / "runtime")
    router = TenantRuntimeRouter(
        tmp_path,
        runtime_factory=lambda _path, **_kwargs: runtime,  # type: ignore[arg-type]
    )
    router.runtime_for("alpha")

    with pytest.raises(BaseExceptionGroup, match="tenant runtimes failed to close"):
        router.close()

    with pytest.raises(TenantRuntimeRouterClosedError, match="closing"):
        router.lease_for("alpha")
    router.close()
    router.close()

    assert runtime.close_calls == 2
    with pytest.raises(TenantRuntimeRouterClosedError, match="closing"):
        router.lease_for("alpha")


def test_tenant_router_closes_remaining_runtimes_after_base_exception(
    tmp_path: Path,
) -> None:
    class InterruptOnceRuntime(_FakeRuntime):
        def close(self) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                raise KeyboardInterrupt("interrupt first runtime close")

    first = InterruptOnceRuntime(tmp_path / "first")
    second = _FakeRuntime(tmp_path / "second")
    created = iter((first, second))
    router = TenantRuntimeRouter(
        tmp_path,
        max_tenant_runtimes=2,
        runtime_factory=lambda _path, **_kwargs: next(created),  # type: ignore[arg-type]
    )
    router.runtime_for("alpha")
    router.runtime_for("beta")

    with pytest.raises(BaseExceptionGroup, match="tenant runtimes failed to close"):
        router.close()

    assert [first.close_calls, second.close_calls] == [1, 1]

    router.close()

    assert [first.close_calls, second.close_calls] == [2, 1]


def test_tenant_router_releases_shutdown_claim_when_condition_exit_is_interrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeRuntime(tmp_path / "runtime")
    router = TenantRuntimeRouter(
        tmp_path,
        runtime_factory=lambda _path, **_kwargs: runtime,  # type: ignore[arg-type]
    )
    router.runtime_for("alpha")
    condition_type = type(router._condition)
    original_exit = condition_type.__exit__
    interrupt_exit = True

    def interrupted_exit(
        condition: threading.Condition,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> bool | None:
        nonlocal interrupt_exit
        result = original_exit(condition, exc_type, exc_value, traceback)
        if condition is router._condition and interrupt_exit:
            interrupt_exit = False
            raise KeyboardInterrupt
        return result

    monkeypatch.setattr(condition_type, "__exit__", interrupted_exit)
    with pytest.raises(KeyboardInterrupt):
        router.close()
    monkeypatch.setattr(condition_type, "__exit__", original_exit)

    retry_completed = threading.Event()
    retry_failures: list[BaseException] = []

    def retry_close() -> None:
        try:
            router.close()
        except BaseException as exc:
            retry_failures.append(exc)
        finally:
            retry_completed.set()

    retry = threading.Thread(target=retry_close, daemon=True)
    retry.start()
    completed_without_repair = retry_completed.wait(timeout=2)
    if not completed_without_repair:
        # Unblock a defective implementation so the failed regression leaves no thread.
        with router._condition:
            router._shutdown_in_flight = False
            router._condition.notify_all()
    retry.join(timeout=2)

    assert completed_without_repair
    assert retry_failures == []
    assert runtime.close_calls == 1


def test_concurrent_shutdown_callers_close_each_runtime_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ConcurrentCloseRuntime(_FakeRuntime):
        def __init__(self, identity: Path) -> None:
            super().__init__(identity)
            self.close_barrier = threading.Barrier(2)

        def close(self) -> None:
            self.close_calls += 1
            with suppress(threading.BrokenBarrierError):
                self.close_barrier.wait(timeout=0.5)

    runtime = ConcurrentCloseRuntime(tmp_path / "runtime")
    router = TenantRuntimeRouter(
        tmp_path,
        runtime_factory=lambda _path, **_kwargs: runtime,  # type: ignore[arg-type]
    )
    lease = router.lease_for("alpha")
    waiting_threads: set[int] = set()
    waiters_lock = threading.Lock()
    both_waiting = threading.Event()
    original_wait = router._condition.wait

    def observed_wait(timeout: float | None = None) -> bool:
        with waiters_lock:
            waiting_threads.add(threading.get_ident())
            if len(waiting_threads) == 2:
                both_waiting.set()
        return original_wait(timeout)

    monkeypatch.setattr(router._condition, "wait", observed_wait)
    failures: list[BaseException] = []

    def close_router() -> None:
        try:
            router.close()
        except BaseException as exc:
            failures.append(exc)

    closers = [threading.Thread(target=close_router) for _ in range(2)]
    for closer in closers:
        closer.start()
    assert both_waiting.wait(timeout=2)
    assert runtime.close_calls == 0

    lease.release()
    for closer in closers:
        closer.join(timeout=2)
        assert not closer.is_alive()

    assert failures == []
    assert runtime.close_calls == 1


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
