from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from jacobian.canonical import canonicalize_json
from jacobian.contracts.claims import ClaimSpec
from jacobian.contracts.discovery import ExperimentState
from jacobian.contracts.evidence import WitnessRole
from jacobian.contracts.plugins import PluginManifest
from jacobian.contracts.search import (
    SearchArchivePage,
    SearchBudget,
    SearchCheckpoint,
    SearchExperimentSnapshot,
    SearchRunRequest,
    SearchStopReason,
)
from jacobian.runtime import create_runtime
from jacobian.runtime.model import JacobianRuntime
from jacobian.search import SearchError
from jacobian.storage.errors import StorageError
from jacobian.storage.models import StorageLimits


def _install_search_plugin(
    runtime: JacobianRuntime,
    *,
    proposer_entrypoint: str = (
        "tests.component.plugins._fixture_plugins:propose_fixture_values"
    ),
    refiner_entrypoint: str = (
        "tests.component.plugins._fixture_plugins:refine_fixture_search"
    ),
    include_witness_oracle: bool = False,
) -> tuple[str, str]:
    claim_schema_uri = runtime.core.schemas.register(
        name="fixture.search-claim",
        version="1",
        schema=ClaimSpec.model_json_schema(),
    )
    candidate_schema_uri = runtime.core.schemas.register(
        name="fixture.search-candidate",
        version="1",
        schema={
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
            "additionalProperties": False,
        },
    )
    semantics_uri = runtime.core.store.register_descriptor(
        kind="semantics",
        name="fixture.search-domain",
        version="1",
        definition={"description": "finite integer search fixture"},
    )
    entrypoints = {
        "Proposer": proposer_entrypoint,
        "Refiner": refiner_entrypoint,
        "Evaluator": "tests.component.plugins._fixture_plugins:evaluate_candidate",
    }
    if include_witness_oracle:
        entrypoints["WitnessOracle"] = (
            "tests.component.plugins._fixture_plugins:find_fixture_witness"
        )
    capabilities: dict[str, dict[str, str]] = {}
    for name, entrypoint in entrypoints.items():
        capabilities[name] = {
            "implementation_uri": runtime.core.plugins.register_implementation(
                entrypoint
            ),
            "entrypoint": entrypoint,
            "version": "1",
        }
    manifest = runtime.core.artifacts.put(
        schema_uri=runtime.services.reference_installer.manifest_schema_uri,
        semantics_uri=runtime.services.reference_installer.manifest_semantics_uri,
        payload=PluginManifest(
            domain_id="fixture.search-domain",
            domain_version="1",
            semantics_uri=semantics_uri,
            claim_schema_uri=claim_schema_uri,
            candidate_schema_uri=candidate_schema_uri,
            capabilities=capabilities,
        ).model_dump(mode="json"),
    )
    runtime.core.plugins.install(manifest.artifact_uri)
    claim = runtime.core.artifacts.put(
        schema_uri=claim_schema_uri,
        semantics_uri=semantics_uri,
        payload={
            "claim_schema_version": "1",
            "domain_id": "fixture.search-domain",
            "domain_version": "1",
            "semantics_uri": semantics_uri,
            "quantifiers": [],
            "predicate": {"name": "fixture_predicate", "parameters": {}},
            "bounds": {},
            "required_capabilities": ["Proposer", "Refiner", "Evaluator"],
            "correspondence_status": "UNREVIEWED",
        },
    )
    return claim.artifact_uri, manifest.artifact_uri


def _request(
    claim_uri: str,
    plugin_id: str,
    *,
    idempotency_key: str,
    batch_size: int = 1,
    wall_seconds: int = 30,
    witness_role: WitnessRole | None = None,
    counterexample_checker_id: str | None = None,
) -> SearchRunRequest:
    return SearchRunRequest(
        idempotency_key=idempotency_key,
        claim_uri=claim_uri,
        plugin_id=plugin_id,
        initial_state={"cursor": 0},
        witness_role=witness_role,
        counterexample_checker_id=counterexample_checker_id,
        budget=SearchBudget(
            candidates_max=8,
            iterations_max=8,
            wall_seconds=wall_seconds,
            batch_size=batch_size,
            workers=1,
        ),
    )


def test_search_run_checkpoints_strategy_neutral_lineage(
    fresh_complete_runtime,
) -> None:
    claim_uri, plugin_id = _install_search_plugin(fresh_complete_runtime)

    handle = fresh_complete_runtime.services.search.start(
        _request(
            claim_uri,
            plugin_id,
            idempotency_key="search-lineage-001",
            batch_size=4,
        )
    )
    snapshot = fresh_complete_runtime.services.search.wait(
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
        fresh_complete_runtime.core.store.get(snapshot.archive_uri).manifest.parents
    ) == {
        claim_uri,
        plugin_id,
        snapshot.checkpoint_uri,
    }
    events = fresh_complete_runtime.services.search.events(handle.experiment_uri)
    assert events[0].event_type == "REQUEST_ACCEPTED"
    assert events[-1].event_type == "COMPLETED"
    proposer_event = next(
        event for event in events if event.event_type == "PROPOSER_COMPLETED"
    )
    assert proposer_event.payload["request_digest"].startswith("sha256:")
    assert proposer_event.payload["output_digest"].startswith("sha256:")


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


def test_concurrent_retries_create_one_search_invocation(
    fresh_complete_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim_uri, plugin_id = _install_search_plugin(fresh_complete_runtime)
    request = _request(
        claim_uri,
        plugin_id,
        idempotency_key="search-concurrent-001",
        batch_size=4,
    )

    with ThreadPoolExecutor(max_workers=8) as pool:
        handles = tuple(
            pool.map(
                lambda _index: fresh_complete_runtime.services.search.start(request),
                range(8),
            )
        )

    experiment_uris = {handle.experiment_uri for handle in handles}
    assert len(experiment_uris) == 1
    experiment_uri = experiment_uris.pop()
    snapshot = fresh_complete_runtime.services.search.wait(
        experiment_uri, timeout_seconds=30
    )
    assert snapshot.accounting.proposed_candidates == 4

    def fail_if_resolved(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("durable retries must not re-resolve plugin code")

    monkeypatch.setattr(
        fresh_complete_runtime.core.plugins, "resolve", fail_if_resolved
    )
    retried = fresh_complete_runtime.services.search.start(request)
    assert retried.experiment_uri == experiment_uri

    event_types = [
        event.event_type
        for event in fresh_complete_runtime.services.search.events(experiment_uri)
    ]
    assert event_types.count("REQUEST_ACCEPTED") == 1
    assert event_types.count("REQUEST_REUSED") == 8


def test_search_lifecycle_events_are_append_only(fresh_complete_runtime) -> None:
    claim_uri, plugin_id = _install_search_plugin(fresh_complete_runtime)
    handle = fresh_complete_runtime.services.search.start(
        _request(
            claim_uri,
            plugin_id,
            idempotency_key="search-events-001",
            batch_size=4,
        )
    )
    fresh_complete_runtime.services.search.wait(
        handle.experiment_uri, timeout_seconds=30
    )

    with (
        sqlite3.connect(fresh_complete_runtime.core.store.db_path) as connection,
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


def test_idempotency_key_cannot_be_rebound(fresh_complete_runtime) -> None:
    claim_uri, plugin_id = _install_search_plugin(fresh_complete_runtime)
    first = _request(
        claim_uri,
        plugin_id,
        idempotency_key="search-rebind-001",
    )
    fresh_complete_runtime.services.search.start(first)

    with pytest.raises(
        SearchError,
        match=(
            r"This idempotency key is already bound to a different request\. "
            r"Reuse the original request or choose a new idempotency key\."
        ),
    ):
        fresh_complete_runtime.services.search.start(
            first.model_copy(
                update={
                    "initial_state": {"cursor": 2},
                }
            )
        )


def test_search_pauses_and_resumes_without_duplicate_lineage(
    fresh_complete_runtime,
) -> None:
    claim_uri, plugin_id = _install_search_plugin(
        fresh_complete_runtime,
        proposer_entrypoint=(
            "tests.component.plugins._fixture_plugins:propose_fixture_values_slowly"
        ),
    )
    handle = fresh_complete_runtime.services.search.start(
        _request(
            claim_uri,
            plugin_id,
            idempotency_key="search-pause-001",
        )
    )

    pause = fresh_complete_runtime.services.search.pause(handle.experiment_uri)
    assert pause.accepted is True
    paused = fresh_complete_runtime.services.search.wait(
        handle.experiment_uri, timeout_seconds=30
    )
    assert paused.state is ExperimentState.PAUSED
    before_pages = paused.archive_page_uris

    resumed = fresh_complete_runtime.services.search.resume(handle.experiment_uri)
    assert resumed.accepted is True
    completed = fresh_complete_runtime.services.search.wait(
        handle.experiment_uri, timeout_seconds=30
    )

    assert completed.state is ExperimentState.COMPLETED
    assert completed.accounting.unique_candidates == 4
    assert len(completed.archive_page_uris) == 4
    assert completed.archive_page_uris[: len(before_pages)] == before_pages
    assert len(set(completed.archive_page_uris)) == 4


def test_interrupted_search_recovers_from_checkpoint_without_chat_state(
    tmp_path: Path,
) -> None:
    runtime = create_runtime(tmp_path)
    claim_uri, plugin_id = _install_search_plugin(
        runtime,
        proposer_entrypoint=(
            "tests.component.plugins._fixture_plugins:propose_fixture_values_slowly"
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
            "tests.component.plugins._fixture_plugins:propose_fixture_values_slowly"
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


def test_proposer_timeout_fails_closed(fresh_complete_runtime) -> None:
    claim_uri, plugin_id = _install_search_plugin(
        fresh_complete_runtime,
        proposer_entrypoint=(
            "tests.component.plugins._fixture_plugins:propose_search_forever"
        ),
    )
    handle = fresh_complete_runtime.services.search.start(
        _request(
            claim_uri,
            plugin_id,
            idempotency_key="search-timeout-001",
            wall_seconds=1,
        )
    )

    with pytest.raises(
        TimeoutError,
        match="Inspect the experiment or wait again with a larger timeout",
    ):
        fresh_complete_runtime.services.search.wait(
            handle.experiment_uri, timeout_seconds=0
        )

    snapshot = fresh_complete_runtime.services.search.wait(
        handle.experiment_uri, timeout_seconds=10
    )

    assert snapshot.state is ExperimentState.TIMEOUT
    assert snapshot.stop_reason is SearchStopReason.WALL_TIME_LIMIT
    assert snapshot.verification.value == "UNVERIFIED"
    assert snapshot.accounting.proposed_candidates == 0
    assert snapshot.accounting.wall_time_ms > 0
    assert snapshot.archive_page_uris == ()


def test_malformed_proposal_fails_without_evidence_promotion(
    fresh_complete_runtime,
) -> None:
    claim_uri, plugin_id = _install_search_plugin(
        fresh_complete_runtime,
        proposer_entrypoint=(
            "tests.component.plugins._fixture_plugins:propose_malformed_search"
        ),
    )
    handle = fresh_complete_runtime.services.search.start(
        _request(
            claim_uri,
            plugin_id,
            idempotency_key="search-malformed-001",
        )
    )

    snapshot = fresh_complete_runtime.services.search.wait(
        handle.experiment_uri, timeout_seconds=15
    )

    assert snapshot.state is ExperimentState.ERROR
    assert snapshot.verification.value == "UNVERIFIED"
    assert "artifact or plugin response was invalid" in snapshot.detail
    assert "reference contract" in snapshot.detail
    assert "input_value" not in snapshot.detail
    assert snapshot.archive_page_uris == ()


def test_partial_iteration_accounting_survives_malformed_candidate(
    fresh_complete_runtime,
) -> None:
    claim_uri, plugin_id = _install_search_plugin(
        fresh_complete_runtime,
        proposer_entrypoint=(
            "tests.component.plugins._fixture_plugins:propose_partially_invalid_search"
        ),
    )
    handle = fresh_complete_runtime.services.search.start(
        _request(
            claim_uri,
            plugin_id,
            idempotency_key="search-partial-accounting-001",
            batch_size=2,
        )
    )

    snapshot = fresh_complete_runtime.services.search.wait(
        handle.experiment_uri, timeout_seconds=15
    )

    assert snapshot.state is ExperimentState.ERROR
    assert snapshot.accounting.proposed_candidates == 1
    assert snapshot.accounting.unique_candidates == 1
    assert snapshot.accounting.evaluated_candidates == 0


@pytest.mark.parametrize(
    ("entrypoint", "detail", "case_id"),
    [
        (
            "tests.component.plugins._fixture_plugins:propose_declared_failure",
            (
                "The plugin stopped before returning a result. Retry once; "
                "if it happens again, inspect the local plugin log."
            ),
            "declared",
        ),
        (
            "tests.component.plugins._fixture_plugins:propose_large_search_output",
            "The plugin returned too much data. Retry with a smaller request.",
            "output",
        ),
    ],
)
def test_search_plugin_failures_remain_operational(
    fresh_complete_runtime,
    entrypoint: str,
    detail: str,
    case_id: str,
) -> None:
    if entrypoint.endswith("propose_large_search_output"):
        fresh_complete_runtime.services.plugin_executor.max_output_bytes = 1024
    claim_uri, plugin_id = _install_search_plugin(
        fresh_complete_runtime,
        proposer_entrypoint=entrypoint,
    )
    handle = fresh_complete_runtime.services.search.start(
        _request(
            claim_uri,
            plugin_id,
            idempotency_key=f"search-failure-{case_id}",
        )
    )

    snapshot = fresh_complete_runtime.services.search.wait(
        handle.experiment_uri, timeout_seconds=15
    )

    assert snapshot.state is ExperimentState.ERROR
    assert snapshot.verification.value == "UNVERIFIED"
    assert detail in snapshot.detail
    assert snapshot.archive_page_uris == ()


def test_terminal_archive_failure_marks_search_error(
    fresh_complete_runtime,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    claim_uri, plugin_id = _install_search_plugin(fresh_complete_runtime)

    def fail_archive(*_args: object, **_kwargs: object) -> object:
        raise StorageError("fixture archive failure")

    monkeypatch.setattr(
        fresh_complete_runtime.services.search, "_store_archive", fail_archive
    )
    handle = fresh_complete_runtime.services.search.start(
        _request(
            claim_uri,
            plugin_id,
            idempotency_key="search-terminal-archive-failure-001",
            batch_size=4,
        )
    )

    snapshot = fresh_complete_runtime.services.search.wait(
        handle.experiment_uri, timeout_seconds=15
    )

    assert snapshot.state is ExperimentState.ERROR
    assert snapshot.stop_reason is SearchStopReason.ERROR
    assert snapshot.archive_uri is None
    assert "could not save the final experiment archive" in snapshot.detail
    assert "experiment remains unverified" in snapshot.detail
    assert "StorageError" not in snapshot.detail
    assert "fixture archive failure" not in snapshot.detail
    assert "fixture archive failure" in caplog.text


def test_plugin_cannot_widen_operator_batch_policy(fresh_complete_runtime) -> None:
    fresh_complete_runtime.services.search.max_batch_size = 1
    claim_uri, plugin_id = _install_search_plugin(
        fresh_complete_runtime,
        proposer_entrypoint=(
            "tests.component.plugins._fixture_plugins:propose_beyond_authority"
        ),
    )
    handle = fresh_complete_runtime.services.search.start(
        _request(
            claim_uri,
            plugin_id,
            idempotency_key="search-policy-001",
            batch_size=8,
        )
    )

    snapshot = fresh_complete_runtime.services.search.wait(
        handle.experiment_uri, timeout_seconds=15
    )

    assert snapshot.effective_budget.batch_size == 1
    assert snapshot.state is ExperimentState.ERROR
    assert "more candidates than authorized" in snapshot.detail
    assert snapshot.accounting.proposed_candidates == 0


def test_search_batch_respects_evaluator_limit(fresh_complete_runtime) -> None:
    fresh_complete_runtime.services.evaluation.max_batch_size = 2
    fresh_complete_runtime.services.search.max_batch_size = 3
    claim_uri, plugin_id = _install_search_plugin(fresh_complete_runtime)
    handle = fresh_complete_runtime.services.search.start(
        _request(
            claim_uri,
            plugin_id,
            idempotency_key="search-evaluator-batch-policy-001",
            batch_size=3,
        )
    )

    snapshot = fresh_complete_runtime.services.search.wait(
        handle.experiment_uri, timeout_seconds=30
    )

    assert snapshot.state is ExperimentState.COMPLETED
    assert snapshot.effective_budget.batch_size == 2
    assert snapshot.accounting.unique_candidates == 4
    assert snapshot.accounting.iterations == 2


def test_search_batch_respects_archive_parent_limit(fresh_complete_runtime) -> None:
    claim_uri, plugin_id = _install_search_plugin(fresh_complete_runtime)
    fresh_complete_runtime.core.store.limits = StorageLimits(max_parents=6)
    handle = fresh_complete_runtime.services.search.start(
        _request(
            claim_uri,
            plugin_id,
            idempotency_key="search-archive-parent-policy-001",
            batch_size=4,
        )
    )

    snapshot = fresh_complete_runtime.services.search.wait(
        handle.experiment_uri, timeout_seconds=30
    )

    assert snapshot.state is ExperimentState.COMPLETED
    assert snapshot.effective_budget.batch_size == 3
    assert snapshot.accounting.unique_candidates == 4
    assert snapshot.accounting.iterations == 2
    for page_uri in snapshot.archive_page_uris:
        assert (
            len(fresh_complete_runtime.core.store.get(page_uri).manifest.parents) <= 6
        )


def test_refiner_cannot_claim_verification(fresh_complete_runtime) -> None:
    claim_uri, plugin_id = _install_search_plugin(
        fresh_complete_runtime,
        refiner_entrypoint=(
            "tests.component.plugins._fixture_plugins:refine_with_verification_claim"
        ),
    )
    handle = fresh_complete_runtime.services.search.start(
        _request(
            claim_uri,
            plugin_id,
            idempotency_key="search-promotion-001",
            batch_size=4,
        )
    )

    snapshot = fresh_complete_runtime.services.search.wait(
        handle.experiment_uri, timeout_seconds=15
    )

    assert snapshot.state is ExperimentState.ERROR
    assert snapshot.verification.value == "UNVERIFIED"
    assert "artifact or plugin response was invalid" in snapshot.detail
    assert "verification" not in snapshot.detail
    assert snapshot.archive_page_uris == ()


def test_verified_counterexample_feedback_reaches_refiner(
    fresh_complete_runtime,
) -> None:
    claim_uri, plugin_id = _install_search_plugin(
        fresh_complete_runtime,
        refiner_entrypoint=(
            "tests.component.plugins._fixture_plugins:refine_from_verified_counterexample"
        ),
        include_witness_oracle=True,
    )
    manifest = fresh_complete_runtime.core.plugins.get(plugin_id)
    checker = fresh_complete_runtime.core.checkers.authorize(
        name="fixture-value-v1",
        entrypoint="tests.component.checkers._fixture_checkers:check_fixture_value",
        evidence_kind="WITNESS",
        format_id="fixture.value",
        format_version="1",
        claim_schema_uris=(manifest.claim_schema_uri,),
        semantics_uris=(manifest.semantics_uri,),
        candidate_schema_uris=(manifest.candidate_schema_uri,),
        reason="search orchestration conformance fixture",
    )
    fresh_complete_runtime.core.store.limits = StorageLimits(max_parents=9)
    handle = fresh_complete_runtime.services.search.start(
        _request(
            claim_uri,
            plugin_id,
            idempotency_key="search-feedback-001",
            batch_size=4,
            witness_role=WitnessRole.DEFEATS_CANDIDATE,
            counterexample_checker_id=checker.checker_id,
        )
    )

    snapshot = fresh_complete_runtime.services.search.wait(
        handle.experiment_uri, timeout_seconds=30
    )

    assert snapshot.state is ExperimentState.COMPLETED
    assert snapshot.effective_budget.batch_size == 2
    assert snapshot.accounting.iterations == 2
    assert snapshot.accounting.attacked_candidates == 4
    assert snapshot.accounting.verified_counterexamples == 4
    assert snapshot.checkpoint_uri is not None
    checkpoint = SearchCheckpoint.model_validate(
        fresh_complete_runtime.core.store.get(snapshot.checkpoint_uri).payload
    )
    assert checkpoint.state["saw_verified_counterexample"] is True
    assert all(record.counterexample_verified for record in checkpoint.latest_records)
    assert all(
        record.verification_record_uri is not None
        for record in checkpoint.latest_records
    )


def test_supporting_checker_decision_is_not_counted_as_counterexample(
    fresh_complete_runtime,
) -> None:
    claim_uri, plugin_id = _install_search_plugin(
        fresh_complete_runtime,
        include_witness_oracle=True,
    )
    manifest = fresh_complete_runtime.core.plugins.get(plugin_id)
    checker = fresh_complete_runtime.core.checkers.authorize(
        name="fixture-value-true-v1",
        entrypoint=(
            "tests.component.checkers._fixture_checkers:check_fixture_value_as_true"
        ),
        evidence_kind="WITNESS",
        format_id="fixture.value",
        format_version="1",
        claim_schema_uris=(manifest.claim_schema_uri,),
        semantics_uris=(manifest.semantics_uri,),
        candidate_schema_uris=(manifest.candidate_schema_uri,),
        reason="counterexample conclusion boundary fixture",
    )
    handle = fresh_complete_runtime.services.search.start(
        _request(
            claim_uri,
            plugin_id,
            idempotency_key="search-supporting-decision-001",
            batch_size=4,
            witness_role=WitnessRole.DEFEATS_CANDIDATE,
            counterexample_checker_id=checker.checker_id,
        )
    )

    snapshot = fresh_complete_runtime.services.search.wait(
        handle.experiment_uri, timeout_seconds=30
    )

    assert snapshot.state is ExperimentState.COMPLETED
    assert snapshot.accounting.verified_counterexamples == 0
    assert snapshot.checkpoint_uri is not None
    checkpoint = SearchCheckpoint.model_validate(
        fresh_complete_runtime.core.store.get(snapshot.checkpoint_uri).payload
    )
    assert all(
        not record.counterexample_verified for record in checkpoint.latest_records
    )
