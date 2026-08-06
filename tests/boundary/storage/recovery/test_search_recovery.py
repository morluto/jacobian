"""Search orchestration recovery and checkpoint boundary tests.

Covers: resume rejects archive pages rebound to another plugin, checkpoint
persistence wall-time accounting, budget enforcement at checkpoint write,
crash recovery from a paused checkpoint, cancellation recovery after process
loss, and corrupt/mismatched snapshot quarantine without blocking valid
recovery.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from search_orchestration_support import _install_search_plugin, _request

from jacobian.canonical import canonicalize_json
from jacobian.contracts.discovery import ExperimentState
from jacobian.contracts.search import (
    SearchArchivePage,
    SearchExperimentSnapshot,
    SearchStopReason,
)
from jacobian.runtime import create_runtime


def test_resume_rejects_archive_page_rebound_to_another_plugin(
    fresh_complete_runtime,
) -> None:
    claim_uri, plugin_id = _install_search_plugin(fresh_complete_runtime)
    handle = fresh_complete_runtime.services.search.start(
        _request(
            claim_uri,
            plugin_id,
            idempotency_key="search-rebound-page-001",
            batch_size=4,
        )
    )
    completed = fresh_complete_runtime.services.search.wait(
        handle.experiment_uri, timeout_seconds=30
    )
    original_page = fresh_complete_runtime.core.store.get(
        completed.archive_page_uris[0]
    )
    rebound_page = SearchArchivePage.model_validate(original_page.payload).model_copy(
        update={"plugin_id": claim_uri}
    )
    stored_rebound_page = fresh_complete_runtime.services.search._put_internal_artifact(
        schema_uri=fresh_complete_runtime.services.search.archive_page_schema_uri,
        payload=rebound_page.model_dump(mode="json"),
        parents=original_page.manifest.parents,
        summary="search archive page",
    )
    paused = completed.model_copy(
        update={
            "state": ExperimentState.PAUSED,
            "stop_reason": None,
            "strategy_reported_complete": False,
            "archive_uri": None,
            "archive_page_uris": (stored_rebound_page.artifact_uri,),
        }
    )
    with sqlite3.connect(fresh_complete_runtime.core.store.db_path) as connection:
        connection.execute(
            """
            UPDATE search_experiments
            SET state = ?, snapshot_json = ?
            WHERE experiment_uri = ?
            """,
            (
                paused.state.value,
                canonicalize_json(paused.model_dump(mode="json")),
                paused.experiment_uri,
            ),
        )

    resumed = fresh_complete_runtime.services.search.resume(handle.experiment_uri)
    assert resumed.accepted is True
    recovered = fresh_complete_runtime.services.search.wait(
        handle.experiment_uri, timeout_seconds=30
    )

    assert recovered.state is ExperimentState.ERROR
    assert "archive page identity does not match the search" in recovered.detail


def test_checkpoint_persistence_is_included_in_wall_accounting(
    fresh_complete_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim_uri, plugin_id = _install_search_plugin(fresh_complete_runtime)
    original_put = fresh_complete_runtime.services.search._put_internal_artifact
    current_time = 0.0

    def clock() -> float:
        return current_time

    def delayed_put(**kwargs: object) -> object:
        nonlocal current_time
        if (
            kwargs.get("schema_uri")
            == fresh_complete_runtime.services.search.checkpoint_schema_uri
        ):
            current_time += 1
        return original_put(**kwargs)

    monkeypatch.setattr(fresh_complete_runtime.services.search, "_clock", clock)
    monkeypatch.setattr(
        fresh_complete_runtime.services.search, "_put_internal_artifact", delayed_put
    )
    handle = fresh_complete_runtime.services.search.start(
        _request(
            claim_uri,
            plugin_id,
            idempotency_key="search-accounting-persistence-001",
            batch_size=4,
        )
    )
    snapshot = fresh_complete_runtime.services.search.wait(
        handle.experiment_uri, timeout_seconds=30
    )

    assert snapshot.state is ExperimentState.COMPLETED
    assert snapshot.accounting.wall_time_ms >= 1_000


def test_checkpoint_persistence_cannot_complete_past_wall_budget(
    fresh_complete_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim_uri, plugin_id = _install_search_plugin(fresh_complete_runtime)
    original_put = fresh_complete_runtime.services.search._put_internal_artifact
    current_time = 0.0

    def clock() -> float:
        return current_time

    def delayed_put(**kwargs: object) -> object:
        nonlocal current_time
        if (
            kwargs.get("schema_uri")
            == fresh_complete_runtime.services.search.checkpoint_schema_uri
        ):
            current_time += 5.1
        return original_put(**kwargs)

    monkeypatch.setattr(fresh_complete_runtime.services.search, "_clock", clock)
    monkeypatch.setattr(
        fresh_complete_runtime.services.search, "_put_internal_artifact", delayed_put
    )
    handle = fresh_complete_runtime.services.search.start(
        _request(
            claim_uri,
            plugin_id,
            idempotency_key="search-accounting-timeout-001",
            batch_size=4,
            wall_seconds=5,
        )
    )
    snapshot = fresh_complete_runtime.services.search.wait(
        handle.experiment_uri, timeout_seconds=30
    )

    assert snapshot.state is ExperimentState.TIMEOUT
    assert snapshot.stop_reason is SearchStopReason.WALL_TIME_LIMIT
    assert snapshot.strategy_reported_complete is False
    assert snapshot.accounting.wall_time_ms >= 5_000


def test_interrupted_search_recovers_from_checkpoint_without_chat_state(
    tmp_path: Path,
) -> None:
    runtime = create_runtime(tmp_path)
    claim_uri, plugin_id = _install_search_plugin(
        runtime,
        proposer_entrypoint=(
            "tests.support.search_entrypoints:propose_fixture_values_slowly"
        ),
    )
    handle = runtime.services.search.start(
        _request(
            claim_uri,
            plugin_id,
            idempotency_key="search-recovery-001",
        )
    )
    runtime.services.search.pause(handle.experiment_uri)
    paused = runtime.services.search.wait(handle.experiment_uri, timeout_seconds=30)
    thread = runtime.services.search._threads.get(handle.experiment_uri)
    if thread is not None:
        thread.join(timeout=5)

    simulated_running = SearchExperimentSnapshot.model_validate(
        {
            **paused.model_dump(mode="json"),
            "state": "RUNNING",
            "detail": "simulated process loss",
        }
    )
    with sqlite3.connect(runtime.core.store.db_path) as connection:
        connection.execute(
            """
            UPDATE search_experiments
            SET state = ?, snapshot_json = ?
            WHERE experiment_uri = ?
            """,
            (
                ExperimentState.RUNNING.value,
                canonicalize_json(simulated_running.model_dump(mode="json")),
                handle.experiment_uri,
            ),
        )

    recovered_runtime = create_runtime(tmp_path)
    recovered = recovered_runtime.services.search.inspect(handle.experiment_uri)
    assert recovered.state is ExperimentState.PAUSED
    assert recovered.checkpoint_uri == paused.checkpoint_uri
    recovered_runtime.services.search.resume(handle.experiment_uri)
    completed = recovered_runtime.services.search.wait(
        handle.experiment_uri,
        timeout_seconds=30,
    )

    assert completed.state is ExperimentState.COMPLETED
    assert completed.accounting.unique_candidates == 4
    assert len(set(completed.archive_page_uris)) == 4


def test_interrupted_cancellation_remains_cancelled_after_recovery(
    tmp_path: Path,
) -> None:
    runtime = create_runtime(tmp_path)
    claim_uri, plugin_id = _install_search_plugin(
        runtime,
        proposer_entrypoint=(
            "tests.support.search_entrypoints:propose_fixture_values_slowly"
        ),
    )
    handle = runtime.services.search.start(
        _request(
            claim_uri,
            plugin_id,
            idempotency_key="search-recovery-cancel-001",
        )
    )
    runtime.services.search.pause(handle.experiment_uri)
    paused = runtime.services.search.wait(handle.experiment_uri, timeout_seconds=30)
    thread = runtime.services.search._threads.get(handle.experiment_uri)
    if thread is not None:
        thread.join(timeout=5)

    interrupted = SearchExperimentSnapshot.model_validate(
        {
            **paused.model_dump(mode="json"),
            "state": "CANCEL_REQUESTED",
            "detail": "simulated process loss after cancellation",
        }
    )
    with sqlite3.connect(runtime.core.store.db_path) as connection:
        connection.execute(
            """
            UPDATE search_experiments
            SET state = ?, snapshot_json = ?
            WHERE experiment_uri = ?
            """,
            (
                ExperimentState.CANCEL_REQUESTED.value,
                canonicalize_json(interrupted.model_dump(mode="json")),
                handle.experiment_uri,
            ),
        )

    recovered_runtime = create_runtime(tmp_path)
    recovered = recovered_runtime.services.search.inspect(handle.experiment_uri)

    assert recovered.state is ExperimentState.CANCELLED
    assert recovered.stop_reason is SearchStopReason.CANCELLED
    assert recovered.checkpoint_uri == paused.checkpoint_uri
    assert recovered.archive_uri is not None
    event_types = [
        event.event_type
        for event in recovered_runtime.services.search.events(handle.experiment_uri)
    ]
    assert event_types[-2:] == [
        "RECOVERED_CANCELLED",
        "RECOVERY_ARCHIVE_COMMITTED",
    ]


def test_corrupt_snapshot_is_quarantined_without_blocking_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = create_runtime(tmp_path)
    claim_uri, plugin_id = _install_search_plugin(runtime)
    monkeypatch.setattr(
        runtime.services.search, "_launch", lambda *_args, **_kwargs: None
    )
    valid = runtime.services.search.start(
        _request(
            claim_uri,
            plugin_id,
            idempotency_key="search-recovery-valid-001",
        )
    )
    valid_snapshot = runtime.services.search.inspect(valid.experiment_uri)
    corrupt_uri = "experiment://ffffffffffffffffffffffffffffffff"
    mismatched_uri = "experiment://eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
    invalid_state_uri = "experiment://dddddddddddddddddddddddddddddddd"
    with sqlite3.connect(runtime.core.store.db_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            """
            INSERT INTO search_experiments (
                experiment_uri, state, snapshot_json
            ) VALUES (?, 'RUNNING', ?)
            """,
            (corrupt_uri, b"{"),
        )
        connection.execute(
            """
            INSERT INTO search_experiments (
                experiment_uri, state, snapshot_json
            ) VALUES (?, 'PENDING', ?)
            """,
            (
                mismatched_uri,
                canonicalize_json(valid_snapshot.model_dump(mode="json")),
            ),
        )
        invalid_state_snapshot = valid_snapshot.model_copy(
            update={"experiment_uri": invalid_state_uri}
        )
        connection.execute(
            """
            INSERT INTO search_experiments (
                experiment_uri, state, snapshot_json
            ) VALUES (?, 'BROKEN', ?)
            """,
            (
                invalid_state_uri,
                canonicalize_json(invalid_state_snapshot.model_dump(mode="json")),
            ),
        )

    recovered = create_runtime(tmp_path)

    assert (
        recovered.services.search.inspect(valid.experiment_uri).state
        is ExperimentState.PAUSED
    )
    with sqlite3.connect(recovered.core.store.db_path) as connection:
        states = connection.execute(
            """
            SELECT experiment_uri, state
            FROM search_experiments
            WHERE experiment_uri IN (?, ?, ?)
            ORDER BY experiment_uri
            """,
            (corrupt_uri, mismatched_uri, invalid_state_uri),
        ).fetchall()
        failures = connection.execute(
            """
            SELECT experiment_uri, snapshot_digest, detail
            FROM search_recovery_failures
            WHERE experiment_uri IN (?, ?, ?)
            ORDER BY experiment_uri
            """,
            (corrupt_uri, mismatched_uri, invalid_state_uri),
        ).fetchall()
    assert states == [
        (invalid_state_uri, "ERROR"),
        (mismatched_uri, "ERROR"),
        (corrupt_uri, "ERROR"),
    ]
    assert len(failures) == 3
    assert all(str(failure[1]).startswith("sha256:") for failure in failures)
    assert all("invalid" in str(failure[2]) for failure in failures)
    assert (
        recovered.services.search.events(corrupt_uri)[-1].event_type
        == "RECOVERY_REJECTED"
    )
    assert (
        recovered.services.search.events(mismatched_uri)[-1].event_type
        == "RECOVERY_REJECTED"
    )
    assert (
        recovered.services.search.events(invalid_state_uri)[-1].event_type
        == "RECOVERY_REJECTED"
    )
