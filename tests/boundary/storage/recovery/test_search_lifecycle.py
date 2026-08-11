"""Search orchestration lifecycle and idempotency boundary tests.

Covers: basic run with checkpoint/archive lineage, concurrent idempotency,
append-only event log enforcement, idempotency key rebind rejection, and
pause/resume without duplicate archive lineage.
"""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest
from search_orchestration_support import _install_search_plugin, _request

from jacobian.contracts.discovery import ExperimentState
from jacobian.contracts.search import SearchStopReason
from jacobian.search import SearchError


def test_search_run_checkpoints_strategy_neutral_lineage(
    search_services,
) -> None:
    claim_uri, plugin_id = _install_search_plugin(search_services)

    handle = search_services.application.search.start(
        _request(
            claim_uri,
            plugin_id,
            idempotency_key="search-lineage-001",
            batch_size=4,
        )
    )
    snapshot = search_services.application.search.wait(
        handle.experiment_uri, timeout_seconds=30
    )

    assert snapshot.state is ExperimentState.COMPLETED
    assert snapshot.stop_reason is SearchStopReason.STRATEGY_COMPLETE
    assert snapshot.strategy_reported_complete is True
    assert snapshot.verification.value == "UNVERIFIED"
    assert snapshot.accounting.proposed_candidates == 4
    assert snapshot.accounting.unique_candidates == 4
    assert snapshot.accounting.evaluated_candidates == 4
    assert snapshot.accounting.iterations == 1
    assert snapshot.accounting.checkpoints == 1
    assert snapshot.accounting.nominations == 1
    assert snapshot.effective_budget.workers == 1
    assert len(snapshot.archive_page_uris) == 1
    assert snapshot.checkpoint_uri is not None
    assert snapshot.archive_uri is not None
    assert set(
        search_services.core.store.get(snapshot.archive_uri).manifest.parents
    ) == {
        claim_uri,
        plugin_id,
        snapshot.checkpoint_uri,
    }
    events = search_services.application.search.events(handle.experiment_uri)
    assert events[0].event_type == "REQUEST_ACCEPTED"
    assert events[-1].event_type == "COMPLETED"
    proposer_event = next(
        event for event in events if event.event_type == "PROPOSER_COMPLETED"
    )
    assert proposer_event.payload["request_digest"].startswith("sha256:")
    assert proposer_event.payload["output_digest"].startswith("sha256:")


def test_concurrent_retries_create_one_search_invocation(
    search_services,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim_uri, plugin_id = _install_search_plugin(search_services)
    request = _request(
        claim_uri,
        plugin_id,
        idempotency_key="search-concurrent-001",
        batch_size=4,
    )

    with ThreadPoolExecutor(max_workers=8) as pool:
        handles = tuple(
            pool.map(
                lambda _index: search_services.application.search.start(request),
                range(8),
            )
        )

    experiment_uris = {handle.experiment_uri for handle in handles}
    assert len(experiment_uris) == 1
    experiment_uri = experiment_uris.pop()
    snapshot = search_services.application.search.wait(
        experiment_uri, timeout_seconds=30
    )
    assert snapshot.accounting.proposed_candidates == 4

    def fail_if_resolved(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("durable retries must not re-resolve plugin code")

    monkeypatch.setattr(search_services.core.plugins, "resolve", fail_if_resolved)
    retried = search_services.application.search.start(request)
    assert retried.experiment_uri == experiment_uri

    event_types = [
        event.event_type
        for event in search_services.application.search.events(experiment_uri)
    ]
    assert event_types.count("REQUEST_ACCEPTED") == 1
    assert event_types.count("REQUEST_REUSED") == 8


def test_search_lifecycle_events_are_append_only(search_services) -> None:
    claim_uri, plugin_id = _install_search_plugin(search_services)
    handle = search_services.application.search.start(
        _request(
            claim_uri,
            plugin_id,
            idempotency_key="search-events-001",
            batch_size=4,
        )
    )
    search_services.application.search.wait(handle.experiment_uri, timeout_seconds=30)

    with (
        sqlite3.connect(search_services.core.store.db_path) as connection,
        pytest.raises(
            sqlite3.IntegrityError,
            match="append-only",
        ),
    ):
        connection.execute(
            """
            UPDATE search_events
            SET event_digest = ?
            WHERE experiment_uri = ? AND sequence = 0
            """,
            (
                "sha256:" + "0" * 64,
                handle.experiment_uri,
            ),
        )


def test_idempotency_key_cannot_be_rebound(search_services) -> None:
    claim_uri, plugin_id = _install_search_plugin(search_services)
    first = _request(
        claim_uri,
        plugin_id,
        idempotency_key="search-rebind-001",
    )
    search_services.application.search.start(first)

    with pytest.raises(
        SearchError,
        match=(
            r"This idempotency key is already bound to a different request\. "
            r"Reuse the original request or choose a new idempotency key\."
        ),
    ):
        search_services.application.search.start(
            first.model_copy(
                update={
                    "initial_state": {"cursor": 2},
                }
            )
        )


def test_search_pauses_and_resumes_without_duplicate_lineage(
    search_services,
) -> None:
    claim_uri, plugin_id = _install_search_plugin(
        search_services,
        proposer_entrypoint=(
            "tests.support.search_entrypoints:propose_fixture_values_slowly"
        ),
    )
    handle = search_services.application.search.start(
        _request(
            claim_uri,
            plugin_id,
            idempotency_key="search-pause-001",
        )
    )

    pause = search_services.application.search.pause(handle.experiment_uri)
    assert pause.accepted is True
    paused = search_services.application.search.wait(
        handle.experiment_uri, timeout_seconds=30
    )
    assert paused.state is ExperimentState.PAUSED
    before_pages = paused.archive_page_uris

    resumed = search_services.application.search.resume(handle.experiment_uri)
    assert resumed.accepted is True
    completed = search_services.application.search.wait(
        handle.experiment_uri, timeout_seconds=30
    )

    assert completed.state is ExperimentState.COMPLETED
    assert completed.accounting.unique_candidates == 4
    assert len(completed.archive_page_uris) == 4
    assert completed.archive_page_uris[: len(before_pages)] == before_pages
    assert len(set(completed.archive_page_uris)) == 4
