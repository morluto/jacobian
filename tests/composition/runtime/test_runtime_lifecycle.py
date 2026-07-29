"""Integration coverage of :class:`JacobianRuntime` lifecycle ownership.

These tests pin the runtime ownership contract: explicit ``close``,
idempotence, context-manager semantics, partial-bootstrap failure cleanup,
and use-after-close behavior. The runtime quiesces application-owned workers
before delegating storage teardown to :class:`ArtifactStore`; these tests
verify that ordering and that construction failures release every resource.

The contract is verified through observable use-after-close behavior rather
than private ``_closed`` flags wherever possible.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from jacobian.runtime import create_runtime
from jacobian.runtime.model import RuntimeClosedError
from jacobian.search import SearchError, SearchService
from jacobian.store import StoreClosedError

pytestmark = pytest.mark.usefixtures("attached_complete_runtime")


def test_close_is_idempotent(tmp_path: Path) -> None:
    runtime = create_runtime(tmp_path)

    runtime.close()
    runtime.close()
    with pytest.raises(StoreClosedError):
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
    with pytest.raises(StoreClosedError):
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
    with pytest.raises(StoreClosedError):
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

    def release_after_close_begins() -> None:
        time.sleep(0.05)
        release_worker.set()

    releaser = threading.Thread(target=release_after_close_begins)
    releaser.start()
    runtime.close()
    releaser.join(timeout=1)

    assert worker_finished.is_set()
    assert worker_errors == []
    with pytest.raises(StoreClosedError):
        runtime.core.store.register_descriptor(
            kind="schema",
            name="after-worker-close",
            version="1",
            definition={"type": "object"},
        )


def test_search_close_timeout_keeps_store_open_for_retry(
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
    from jacobian.store import ArtifactStore

    reopened = ArtifactStore(tmp_path)
    reopened.close()


def test_close_failure_keeps_store_closable_on_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression test: JacobianRuntime.close() must set its closed marker only
    # after core.close() succeeds. If service close raises, the runtime
    # must remain closable so a retry can still release the underlying store.
    runtime = create_runtime(tmp_path)
    store = runtime.core.store

    def failing_close(self: object) -> None:
        raise RuntimeError("service close failed")

    monkeypatch.setattr(
        "jacobian.runtime.services.CoreServices.close",
        failing_close,
    )

    with pytest.raises(RuntimeError, match="service close failed"):
        runtime.close()

    # The store was not closed by the failed runtime close. Retrying the
    # runtime close (with the real service close restored) must eventually
    # release the store rather than returning early as already-closed.
    monkeypatch.undo()
    runtime.close()

    with pytest.raises(StoreClosedError), store.connection():
        pass
