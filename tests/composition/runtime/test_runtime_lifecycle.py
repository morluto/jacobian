"""Integration coverage of :class:`JacobianRuntime` lifecycle ownership.

These tests pin the runtime ownership contract: explicit ``close``,
idempotence, context-manager semantics, partial-bootstrap failure cleanup,
and use-after-close behavior. The runtime quiesces application-owned workers
before delegating storage teardown to :class:`ArtifactRepository`; these tests
verify that ordering and that construction failures release every resource.

The contract is verified through observable use-after-close behavior rather
than private ``_closed`` flags wherever possible.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from pydantic import ValidationError

from jacobian.contracts.discovery import ExperimentHandle
from jacobian.contracts.search import ExperimentState
from jacobian.experiments import ExperimentError, ExperimentService
from jacobian.portfolio.result import PortfolioInstallation
from jacobian.runtime import create_runtime
from jacobian.runtime.model import RuntimeClosedError
from jacobian.runtime.services import ApplicationServices, CoreServices
from jacobian.search import SearchError, SearchService
from jacobian.storage.errors import StorageClosedError

# Composition-lane admission category for architecture ratchets.
COMPOSITION_ADMISSION = "LIFECYCLE"


def test_runtime_uses_one_schema_registry_instance(attached_complete_runtime) -> None:
    assert (
        attached_complete_runtime.core.plugins.schemas
        is attached_complete_runtime.core.schemas
    )


def test_close_is_idempotent(tmp_path: Path) -> None:
    runtime = create_runtime(tmp_path)

    runtime.close()
    runtime.close()
    with pytest.raises(StorageClosedError):
        runtime.core.store.register_descriptor(
            kind="schema",
            name="after-double-close",
            version="1",
            definition={"type": "object"},
        )
    with pytest.raises(RuntimeClosedError), runtime:
        pytest.fail("entering a closed runtime must not succeed")


def test_context_manager_closes_runtime(tmp_path: Path) -> None:
    with create_runtime(tmp_path) as runtime:
        # While open the store is usable.
        runtime.core.store.register_descriptor(
            kind="schema",
            name="example.in-context",
            version="1",
            definition={"type": "object"},
        )
    # After the context manager exits the store is closed.
    with pytest.raises(StorageClosedError):
        runtime.core.store.register_descriptor(
            kind="schema",
            name="after-context",
            version="1",
            definition={"type": "object"},
        )


def test_context_manager_exit_suppresses_no_exception(tmp_path: Path) -> None:
    runtime = create_runtime(tmp_path)
    with pytest.raises(ValueError, match="body failure"), runtime:
        raise ValueError("body failure")
    # The store is still closed even though the body raised.
    with pytest.raises(StorageClosedError):
        runtime.core.store.register_descriptor(
            kind="schema",
            name="after-body-failure",
            version="1",
            definition={"type": "object"},
        )


def test_close_quiesces_search_workers_before_closing_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = create_runtime(tmp_path)
    worker_started = threading.Event()
    release_worker = threading.Event()
    worker_finished = threading.Event()
    worker_errors: list[BaseException] = []

    def blocked_run(service: SearchService, _experiment_uri: str) -> None:
        worker_started.set()
        release_worker.wait(timeout=2)
        try:
            service.store.register_descriptor(
                kind="schema",
                name="search-worker-during-close",
                version="1",
                definition={"type": "object"},
            )
        except BaseException as exc:
            worker_errors.append(exc)
        finally:
            worker_finished.set()

    monkeypatch.setattr(SearchService, "_run", blocked_run)
    runtime.services.search._launch("experiment://runtime-close-regression")
    assert worker_started.wait(timeout=1)

    close_entered = threading.Event()
    original_close = runtime.close

    def close_after_signaling() -> None:
        close_entered.set()
        original_close()

    monkeypatch.setattr(runtime, "close", close_after_signaling)

    def release_after_close_begins() -> None:
        close_entered.wait(timeout=2)
        release_worker.set()

    releaser = threading.Thread(target=release_after_close_begins)
    releaser.start()
    runtime.close()
    releaser.join(timeout=1)

    assert worker_finished.is_set()
    assert worker_errors == []
    with pytest.raises(StorageClosedError):
        runtime.core.store.register_descriptor(
            kind="schema",
            name="after-worker-close",
            version="1",
            definition={"type": "object"},
        )


def test_close_quiesces_enumeration_workers_before_closing_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = create_runtime(tmp_path)
    worker_started = threading.Event()
    release_worker = threading.Event()
    worker_errors: list[BaseException] = []

    def blocked_run(service: ExperimentService, _experiment_uri: str) -> None:
        worker_started.set()
        release_worker.wait(timeout=2)
        try:
            service.store.register_descriptor(
                kind="schema",
                name="enumeration-worker-during-close",
                version="1",
                definition={"type": "object"},
            )
        except BaseException as exc:
            worker_errors.append(exc)

    monkeypatch.setattr(ExperimentService, "_run_enumeration", blocked_run)
    runtime.services.experiments._launch_enumeration(
        "experiment://runtime-enumeration-close"
    )
    assert worker_started.wait(timeout=1)

    close_entered = threading.Event()
    original_close = runtime.close

    def close_after_signaling() -> None:
        close_entered.set()
        original_close()

    monkeypatch.setattr(runtime, "close", close_after_signaling)

    releaser = threading.Thread(
        target=lambda: (close_entered.wait(timeout=2), release_worker.set())
    )
    releaser.start()
    runtime.close()
    releaser.join(timeout=1)

    assert worker_errors == []
    with pytest.raises(StorageClosedError):
        runtime.core.store.register_descriptor(
            kind="schema",
            name="after-enumeration-worker-close",
            version="1",
            definition={"type": "object"},
        )


def test_enumeration_close_timeout_keeps_service_closing_until_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = create_runtime(tmp_path)
    worker_started = threading.Event()
    release_worker = threading.Event()

    def blocked_run(_service: ExperimentService, _experiment_uri: str) -> None:
        worker_started.set()
        release_worker.wait(timeout=2)

    monkeypatch.setattr(ExperimentService, "_run_enumeration", blocked_run)
    runtime.services.experiments._launch_enumeration(
        "experiment://runtime-enumeration-timeout"
    )
    assert worker_started.wait(timeout=1)

    with pytest.raises(ExperimentError, match="did not quiesce"):
        runtime.services.experiments.close(timeout_seconds=0)
    with pytest.raises(ExperimentError, match="service is closing"):
        runtime.services.experiments._launch_enumeration(
            "experiment://must-not-launch-after-close-begins"
        )
    runtime.core.store.register_descriptor(
        kind="schema",
        name="store-open-after-enumeration-timeout",
        version="1",
        definition={"type": "object"},
    )
    release_worker.set()
    runtime.close()


def test_negative_search_close_timeout_does_not_enter_closing_state(
    tmp_path: Path,
) -> None:
    runtime = create_runtime(tmp_path)
    try:
        with pytest.raises(ValueError, match="non-negative"):
            runtime.services.search.close(timeout_seconds=-1)
        with pytest.raises(ValidationError):
            runtime.services.search.start({})
    finally:
        runtime.close()


def test_negative_enumeration_close_timeout_does_not_enter_closing_state(
    tmp_path: Path,
) -> None:
    runtime = create_runtime(tmp_path)
    try:
        with pytest.raises(ValueError, match="non-negative"):
            runtime.services.experiments.close(timeout_seconds=-1)
        with pytest.raises(ValidationError):
            runtime.services.experiments.start_enumeration({})
    finally:
        runtime.close()


def test_close_waits_for_a_reserved_search_start_through_worker_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = create_runtime(tmp_path)
    start_reserved = threading.Event()
    release_start = threading.Event()
    start_results: list[ExperimentHandle] = []
    start_errors: list[BaseException] = []
    experiment_uri = "experiment://" + "a" * 32

    def blocked_start(
        service: SearchService,
        _request: object,
    ) -> ExperimentHandle:
        start_reserved.set()
        release_start.wait(timeout=2)
        service._launch(experiment_uri, lifecycle_reserved=True)
        return ExperimentHandle(
            experiment_uri=experiment_uri,
            state=ExperimentState.PENDING,
        )

    monkeypatch.setattr(SearchService, "_start_reserved", blocked_start)
    monkeypatch.setattr(SearchService, "_run", lambda *_args: None)

    def start_search() -> None:
        try:
            start_results.append(runtime.services.search.start({}))
        except BaseException as exc:
            start_errors.append(exc)

    starter = threading.Thread(target=start_search)
    starter.start()
    assert start_reserved.wait(timeout=1)

    close_entered = threading.Event()
    original_close = runtime.close

    def close_after_signaling() -> None:
        close_entered.set()
        original_close()

    monkeypatch.setattr(runtime, "close", close_after_signaling)

    def release_after_close_begins() -> None:
        close_entered.wait(timeout=2)
        release_start.set()

    releaser = threading.Thread(target=release_after_close_begins)
    releaser.start()
    runtime.close()
    starter.join(timeout=1)
    releaser.join(timeout=1)

    assert start_errors == []
    assert start_results == [
        ExperimentHandle(
            experiment_uri=experiment_uri,
            state=ExperimentState.PENDING,
        )
    ]


def test_close_waits_for_a_reserved_enumeration_start_through_worker_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = create_runtime(tmp_path)
    start_reserved = threading.Event()
    release_start = threading.Event()
    start_results: list[ExperimentHandle] = []
    start_errors: list[BaseException] = []
    experiment_uri = "experiment://" + "b" * 32

    def blocked_start(
        service: ExperimentService,
        _request: object,
    ) -> ExperimentHandle:
        start_reserved.set()
        release_start.wait(timeout=2)
        service._launch_enumeration(experiment_uri, lifecycle_reserved=True)
        return ExperimentHandle(
            experiment_uri=experiment_uri,
            state=ExperimentState.PENDING,
        )

    monkeypatch.setattr(
        ExperimentService,
        "_start_enumeration_reserved",
        blocked_start,
    )
    monkeypatch.setattr(ExperimentService, "_run_enumeration", lambda *_args: None)

    def start_enumeration() -> None:
        try:
            start_results.append(runtime.services.experiments.start_enumeration({}))
        except BaseException as exc:
            start_errors.append(exc)

    starter = threading.Thread(target=start_enumeration)
    starter.start()
    assert start_reserved.wait(timeout=1)
    close_entered = threading.Event()
    original_close = runtime.close

    def close_after_signaling() -> None:
        close_entered.set()
        original_close()

    monkeypatch.setattr(runtime, "close", close_after_signaling)

    releaser = threading.Thread(
        target=lambda: (close_entered.wait(timeout=2), release_start.set())
    )
    releaser.start()
    runtime.close()
    starter.join(timeout=1)
    releaser.join(timeout=1)

    assert start_errors == []
    assert start_results == [
        ExperimentHandle(
            experiment_uri=experiment_uri,
            state=ExperimentState.PENDING,
        )
    ]


def test_search_close_timeout_keeps_service_closing_until_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = create_runtime(tmp_path)
    worker_started = threading.Event()
    release_worker = threading.Event()

    def blocked_run(_service: SearchService, _experiment_uri: str) -> None:
        worker_started.set()
        release_worker.wait(timeout=2)

    monkeypatch.setattr(SearchService, "_run", blocked_run)
    runtime.services.search._launch("experiment://runtime-close-timeout")
    assert worker_started.wait(timeout=1)

    with pytest.raises(SearchError, match="did not quiesce"):
        runtime.services.search.close(timeout_seconds=0)
    with pytest.raises(SearchError, match="service is closing"):
        runtime.services.search._launch(
            "experiment://must-not-launch-after-close-begins"
        )

    runtime.core.store.register_descriptor(
        kind="schema",
        name="store-remains-open-after-search-close-timeout",
        version="1",
        definition={"type": "object"},
    )
    release_worker.set()
    runtime.services.search.close(timeout_seconds=1)
    runtime.close()


def test_partial_initialize_failure_releases_owned_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # If capability installation fails partway through construction, the
    # runtime must still release the store it already owns so that no SQLite
    # lifetime is leaked. The partially constructed runtime is unreachable
    # from the caller, so the proof is that the same root can be reopened
    # cleanly afterwards without a stale transaction-recovery marker.
    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("boom during initialize")

    monkeypatch.setattr("jacobian.portfolio.install_portfolio", boom)

    with pytest.raises(RuntimeError, match="boom during initialize"):
        create_runtime(tmp_path)

    # Undo the failure injection so the reopened runtime can install its
    # portfolio normally.
    monkeypatch.undo()

    # The store must have been closed by the construction-failure cleanup, so a
    # fresh runtime can take ownership of the same root without conflict.
    reopened = create_runtime(tmp_path)
    try:
        reopened.core.store.register_descriptor(
            kind="schema",
            name="after-partial-initialize",
            version="1",
            definition={"type": "object"},
        )
    finally:
        reopened.close()


def test_partial_bootstrap_failure_releases_owned_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # If the foundational service bootstrap fails before the runtime is even
    # constructed, the store it opened must be closed by the bootstrap cleanup.
    def failing_install(*args: object, **kwargs: object) -> None:
        raise RuntimeError("bootstrap failure")

    monkeypatch.setattr(
        "jacobian.runtime.bootstrap.install_sat_artifacts",
        failing_install,
    )

    with pytest.raises(RuntimeError, match="bootstrap failure"):
        create_runtime(tmp_path)

    # A fresh store can reopen the same root cleanly afterwards.
    from jacobian.storage.repository import ArtifactRepository

    reopened = ArtifactRepository(tmp_path)
    reopened.close()


def test_bootstrap_cleanup_failure_preserves_primary_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.storage.repository import ArtifactRepository

    def failing_install(*args: object, **kwargs: object) -> None:
        raise RuntimeError("bootstrap failure")

    original_close = ArtifactRepository.close

    def close_then_fail(store: ArtifactRepository) -> None:
        original_close(store)
        raise OSError("store close failure")

    monkeypatch.setattr(
        "jacobian.runtime.bootstrap.install_sat_artifacts",
        failing_install,
    )
    monkeypatch.setattr(ArtifactRepository, "close", close_then_fail)

    with pytest.raises(RuntimeError, match="bootstrap failure") as caught:
        create_runtime(tmp_path)

    assert caught.value.__notes__ == [
        "service bootstrap cleanup also failed: store close failure"
    ]


def test_close_failure_keeps_store_closable_on_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression test: JacobianRuntime.close() must set its closed marker only
    # after core.close() succeeds. If core close raises, the runtime must remain
    # closable so a retry can still release the underlying store.
    runtime = create_runtime(tmp_path)
    store = runtime.core.store

    def failing_close(self: object) -> None:
        raise RuntimeError("core close failed")

    monkeypatch.setattr(
        "jacobian.runtime.services.CoreServices.close",
        failing_close,
    )

    with pytest.raises(
        ExceptionGroup, match="runtime resources failed to close"
    ) as exc:
        runtime.close()
    assert [str(failure) for failure in exc.value.exceptions] == [
        "core close failed",
    ]

    # The store was not closed by the failed runtime close. Retrying the
    # runtime close (with the real service close restored) must eventually
    # release the store rather than returning early as already-closed.
    monkeypatch.undo()
    runtime.close()

    with pytest.raises(StorageClosedError), store.connection():
        pass


def test_close_attempts_every_owner_before_raising_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = create_runtime(tmp_path)
    store = runtime.core.store
    close_order: list[str] = []
    services_close = ApplicationServices.close
    portfolio_close = PortfolioInstallation.close
    core_close = CoreServices.close

    def fail_after_services_close(self: ApplicationServices) -> None:
        close_order.append("services")
        services_close(self)
        raise RuntimeError("services close failed")

    def fail_after_portfolio_close(self: PortfolioInstallation) -> None:
        close_order.append("portfolio")
        portfolio_close(self)
        raise RuntimeError("portfolio close failed")

    def fail_after_core_close(self: CoreServices) -> None:
        close_order.append("core")
        core_close(self)
        raise RuntimeError("core close failed")

    monkeypatch.setattr(ApplicationServices, "close", fail_after_services_close)
    monkeypatch.setattr(PortfolioInstallation, "close", fail_after_portfolio_close)
    monkeypatch.setattr(CoreServices, "close", fail_after_core_close)

    with pytest.raises(
        ExceptionGroup, match="runtime resources failed to close"
    ) as exc:
        runtime.close()

    assert close_order == ["services", "portfolio", "core"]
    assert [str(failure) for failure in exc.value.exceptions] == [
        "services close failed",
        "portfolio close failed",
        "core close failed",
    ]
    with pytest.raises(StorageClosedError), store.connection():
        pass

    monkeypatch.undo()
    runtime.close()
    runtime.close()


def test_close_attempts_every_owner_after_keyboard_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interrupting one owner must not skip shutdown of the remaining owners."""

    runtime = create_runtime(tmp_path)
    store = runtime.core.store
    close_order: list[str] = []
    services_close = ApplicationServices.close
    portfolio_close = PortfolioInstallation.close
    core_close = CoreServices.close

    def interrupt_after_services_close(self: ApplicationServices) -> None:
        close_order.append("services")
        services_close(self)
        raise KeyboardInterrupt("services close interrupted")

    def record_portfolio_close(self: PortfolioInstallation) -> None:
        close_order.append("portfolio")
        portfolio_close(self)

    def record_core_close(self: CoreServices) -> None:
        close_order.append("core")
        core_close(self)

    monkeypatch.setattr(ApplicationServices, "close", interrupt_after_services_close)
    monkeypatch.setattr(PortfolioInstallation, "close", record_portfolio_close)
    monkeypatch.setattr(CoreServices, "close", record_core_close)

    with pytest.raises(
        BaseExceptionGroup, match="runtime resources failed to close"
    ) as exc:
        runtime.close()

    assert close_order == ["services", "portfolio", "core"]
    assert [str(failure) for failure in exc.value.exceptions] == [
        "services close interrupted",
    ]
    with pytest.raises(StorageClosedError), store.connection():
        pass


def test_application_services_close_continues_after_keyboard_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An interrupt from search must not leave experiment workers running."""

    runtime = create_runtime(tmp_path)
    close_order: list[str] = []

    def interrupt_search_close(self: SearchService) -> None:
        close_order.append("search")
        raise KeyboardInterrupt("search close interrupted")

    def record_experiment_close(self: ExperimentService) -> None:
        close_order.append("experiments")

    monkeypatch.setattr(SearchService, "close", interrupt_search_close)
    monkeypatch.setattr(ExperimentService, "close", record_experiment_close)

    with pytest.raises(
        BaseExceptionGroup, match="application services did not quiesce"
    ) as exc:
        runtime.services.close()

    assert close_order == ["search", "experiments"]
    assert [str(failure) for failure in exc.value.exceptions] == [
        "search close interrupted",
    ]

    monkeypatch.undo()
    runtime.close()
