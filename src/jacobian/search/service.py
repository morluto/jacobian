"""Durable search experiment service."""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from jacobian.canonical import canonicalize_json
from jacobian.claims import ClaimValidationService
from jacobian.contracts.artifacts import ArtifactPutResult
from jacobian.contracts.discovery import ExperimentHandle, ExperimentState
from jacobian.contracts.evaluation import EvaluationBatchResult
from jacobian.contracts.plugins import CapabilityName
from jacobian.contracts.results import (
    ExecutionStatus,
    InputStatus,
    InputValidation,
    Verification,
)
from jacobian.contracts.search import (
    ExperimentControlResult,
    SearchAccounting,
    SearchArchiveManifest,
    SearchArchivePage,
    SearchBudget,
    SearchCheckpoint,
    SearchExperimentSnapshot,
    SearchLifecycleEvent,
    SearchRunRequest,
    SearchStopReason,
)
from jacobian.evaluation import (
    EvaluationService,
)
from jacobian.experiment_identity import new_experiment_uri
from jacobian.lifecycle import (
    LifecycleTimeoutError,
    ServiceLifecycleState,
    WorkerLaunchStatus,
    launch_worker,
    wait_for_worker_quiescence,
    wait_for_worker_settlement,
)
from jacobian.persistence import PersistenceCorruptionError, decode_persisted_model
from jacobian.persistence.recovery import (
    put_internal_artifact,
    quarantine_recovery_snapshot,
)
from jacobian.plugin_execution import PluginExecutor
from jacobian.plugins.registry import (
    PluginRegistry,
    PluginRegistryError,
    ResolvedCapability,
)
from jacobian.schema_registry import SchemaRegistry, SchemaRegistryError, model_schema
from jacobian.search._helpers import (
    _digest,
    _environment_digest,
    _event_digest,
    _now,
    _updated_accounting,
    _updated_snapshot,
)
from jacobian.search.errors import SearchCorruptionError, SearchError
from jacobian.storage.errors import StorageError
from jacobian.storage.repository import ArtifactRepository
from jacobian.verification import VerificationService
from jacobian.witnesses import WitnessSearchService

_LOGGER = logging.getLogger(__name__)

_TERMINAL_STATES = {
    ExperimentState.COMPLETED,
    ExperimentState.CANCELLED,
    ExperimentState.TIMEOUT,
    ExperimentState.ERROR,
}
_SETTLED_STATES = _TERMINAL_STATES | {ExperimentState.PAUSED}


class SearchService:
    """Coordinate untrusted strategies over durable verification boundaries.

    SQLite owns idempotent request acceptance, current lifecycle state, and
    append-only events. Immutable artifacts own checkpoints and archive
    lineage. The service may replay work after the last committed checkpoint,
    but it never derives a mathematical conclusion from strategy completion,
    timeout, cancellation, or recovery state.

    The reference scheduler assumes one active coordinator per state directory
    and accepts exactly one strategy worker.
    """

    def __init__(
        self,
        store: ArtifactRepository,
        schemas: SchemaRegistry,
        plugins: PluginRegistry,
        claims: ClaimValidationService,
        executor: PluginExecutor,
        evaluation: EvaluationService,
        witnesses: WitnessSearchService,
        verification: VerificationService,
        *,
        max_candidates: int = 10_000_000,
        max_iterations: int = 10_000_000,
        max_wall_seconds: int = 86_400,
        max_batch_size: int = 256,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.store = store
        self.schemas = schemas
        self.plugins = plugins
        self.claims = claims
        self.executor = executor
        self.evaluation = evaluation
        self.witnesses = witnesses
        self.verification = verification
        self.max_candidates = max_candidates
        self.max_iterations = max_iterations
        self.max_wall_seconds = max_wall_seconds
        self.max_batch_size = max_batch_size
        self._clock = clock
        self._threads: dict[str, threading.Thread] = {}
        self._thread_lock = threading.Lock()
        self._starts_in_flight = 0
        self._lifecycle_state = ServiceLifecycleState.OPEN
        self.semantics_uri = store.register_descriptor(
            kind="semantics",
            name="jacobian.search-experiment",
            version="1",
            definition={
                "description": (
                    "untrusted strategy state, durable progress, and archive lineage"
                )
            },
        )
        self.checkpoint_schema_uri = schemas.register(
            name="jacobian.search-checkpoint",
            version="1",
            schema=model_schema(SearchCheckpoint),
        )
        self.archive_page_schema_uri = schemas.register(
            name="jacobian.search-archive-page",
            version="1",
            schema=model_schema(SearchArchivePage),
        )
        self.archive_manifest_schema_uri = schemas.register(
            name="jacobian.search-archive",
            version="1",
            schema=model_schema(SearchArchiveManifest),
        )
        self.evaluation_schema_uri = schemas.register(
            name="jacobian.evaluation-batch-result",
            version="1",
            schema=model_schema(EvaluationBatchResult),
        )
        self._recover_interrupted_searches()

    def close(self, *, timeout_seconds: float = 30) -> None:
        """Quiesce runtime-owned workers before their shared store is closed."""

        if timeout_seconds < 0:
            raise ValueError("timeout_seconds must be non-negative")
        with self._thread_lock:
            if self._lifecycle_state is ServiceLifecycleState.CLOSED:
                return
            self._lifecycle_state = ServiceLifecycleState.CLOSING
        try:
            wait_for_worker_quiescence(
                lock=self._thread_lock,
                workers=self._threads,
                starts_in_flight=lambda: self._starts_in_flight,
                timeout_seconds=timeout_seconds,
            )
        except LifecycleTimeoutError as exc:
            raise SearchError(
                "search workers did not quiesce before runtime shutdown"
            ) from exc
        with self._thread_lock:
            self._lifecycle_state = ServiceLifecycleState.CLOSED

    def _require_open(self) -> None:
        with self._thread_lock:
            if self._lifecycle_state is not ServiceLifecycleState.OPEN:
                raise SearchError("search service is closing")

    def _recover_interrupted_searches(self) -> None:
        """Isolate interrupted search state one row at a time.

        Active runs recover as paused, pending cancellation recovers as
        cancelled, and malformed or index-inconsistent snapshots are
        quarantined as errors. A corrupt row must not prevent unrelated
        experiments from recovering.
        """

        archive_recoveries: list[SearchExperimentSnapshot] = []
        with self.store.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """
                SELECT experiment_uri, state, snapshot_json
                FROM search_experiments
                WHERE state IN (
                    'PENDING', 'RUNNING', 'PAUSE_REQUESTED',
                    'CANCEL_REQUESTED', 'CANCELLED'
                )
                OR state NOT IN (
                    'PENDING', 'RUNNING', 'PAUSE_REQUESTED', 'PAUSED',
                    'COMPLETED', 'CANCEL_REQUESTED', 'CANCELLED',
                    'TIMEOUT', 'ERROR'
                )
                """
            ).fetchall()
            for row in rows:
                try:
                    snapshot = decode_persisted_model(
                        SearchExperimentSnapshot,
                        row["snapshot_json"],
                        record_kind="search_snapshot",
                        record_id=str(row["experiment_uri"]),
                        field="snapshot_json",
                    )
                except PersistenceCorruptionError as exc:
                    self._quarantine_recovery_snapshot(connection, row, exc)
                    continue
                if snapshot.experiment_uri != str(
                    row["experiment_uri"]
                ) or snapshot.state.value != str(row["state"]):
                    self._quarantine_recovery_snapshot(
                        connection,
                        row,
                        ValueError(
                            "stored search snapshot identity or state differs "
                            "from its database index"
                        ),
                    )
                    continue
                if snapshot.state is ExperimentState.CANCELLED:
                    if snapshot.archive_uri is None:
                        archive_recoveries.append(snapshot)
                    continue
                cancelled = snapshot.state is ExperimentState.CANCEL_REQUESTED
                recovered = _updated_snapshot(
                    snapshot,
                    state=(
                        ExperimentState.CANCELLED
                        if cancelled
                        else ExperimentState.PAUSED
                    ),
                    stop_reason=(SearchStopReason.CANCELLED if cancelled else None),
                    strategy_reported_complete=False,
                    updated_at=_now(),
                    detail=(
                        "cancellation completed during process recovery"
                        if cancelled
                        else (
                            "experiment process ended before completion; "
                            "resume from the last committed checkpoint"
                        )
                    ),
                )
                self._update_snapshot(connection, recovered)
                self._append_event(
                    connection,
                    recovered.experiment_uri,
                    event_type=(
                        "RECOVERED_CANCELLED" if cancelled else "RECOVERED_PAUSED"
                    ),
                    payload={
                        "checkpoint_uri": recovered.checkpoint_uri,
                        "accounting": recovered.accounting.model_dump(mode="json"),
                    },
                )
                if cancelled:
                    archive_recoveries.append(recovered)
        for snapshot in archive_recoveries:
            self._commit_recovery_archive(snapshot)

    def _quarantine_recovery_snapshot(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        error: Exception,
    ) -> None:
        """Isolate one corrupt row without blocking unrelated recovery."""

        detail = (
            "Stored search state is invalid. Restore the Jacobian state directory "
            "from a trusted backup or start a new search."
        )
        snapshot_digest = quarantine_recovery_snapshot(
            connection,
            row,
            error,
            experiments_table="search_experiments",
            recovery_table="search_recovery_failures",
            detail=detail,
            logger=_LOGGER,
            logger_message="quarantining invalid search snapshot for %s",
        )
        experiment_uri = str(row["experiment_uri"])
        try:
            self._append_event(
                connection,
                experiment_uri,
                event_type="RECOVERY_REJECTED",
                payload={
                    "snapshot_digest": snapshot_digest,
                    "detail": detail,
                },
            )
        except ValidationError:
            # The quarantine row remains authoritative when the corrupt
            # experiment identifier cannot itself satisfy the event contract.
            return

    def _commit_recovery_archive(
        self,
        snapshot: SearchExperimentSnapshot,
    ) -> None:
        archive = self._store_archive(snapshot, snapshot.accounting)
        with self.store.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            latest = self._read_snapshot(connection, snapshot.experiment_uri)
            if (
                latest.state is not ExperimentState.CANCELLED
                or latest.archive_uri is not None
            ):
                return
            archived = _updated_snapshot(
                latest,
                archive_uri=archive.artifact_uri,
                updated_at=_now(),
            )
            self._update_snapshot(connection, archived)
            self._append_event(
                connection,
                archived.experiment_uri,
                event_type="RECOVERY_ARCHIVE_COMMITTED",
                payload={"archive_uri": archive.artifact_uri},
            )

    def start(
        self,
        request: SearchRunRequest | dict[str, Any],
    ) -> ExperimentHandle:
        """Commit one idempotent search request and launch it locally."""

        with self._thread_lock:
            if self._lifecycle_state is not ServiceLifecycleState.OPEN:
                raise SearchError("search service is closing")
            self._starts_in_flight += 1
        try:
            return self._start_reserved(request)
        finally:
            with self._thread_lock:
                self._starts_in_flight -= 1

    def _start_reserved(
        self,
        request: SearchRunRequest | dict[str, Any],
    ) -> ExperimentHandle:
        """Start after reserving the service lifecycle through worker launch."""

        selected = SearchRunRequest.model_validate(request)
        request_digest = _digest(selected.model_dump(mode="json"))
        with self.store.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing_handle = self._reuse_request(
                connection,
                selected,
                request_digest,
            )
        if existing_handle is not None:
            return existing_handle

        validation = self.claims.validate(
            claim_uri=selected.claim_uri,
            plugin_id=selected.plugin_id,
        )
        if not validation.valid:
            raise SearchError("; ".join(validation.input.errors))
        try:
            proposer = self.plugins.resolve(
                selected.plugin_id,
                CapabilityName.PROPOSER,
            )
            refiner = self.plugins.resolve(
                selected.plugin_id,
                CapabilityName.REFINER,
            )
            evaluator = self.plugins.resolve(
                selected.plugin_id,
                CapabilityName.EVALUATOR,
            )
            if selected.witness_role is not None:
                witness_oracle = self.plugins.resolve(
                    selected.plugin_id,
                    CapabilityName.WITNESS_ORACLE,
                )
        except PluginRegistryError as exc:
            _LOGGER.warning("search capability resolution failed", exc_info=exc)
            raise SearchError(
                "The search plugin is unavailable or incomplete. Call "
                "math.find, choose a reference domain with proposer, "
                "refiner, and evaluator capabilities, then retry."
            ) from exc

        effective_budget = self._effective_budget(
            selected.budget,
            include_witness_lineage=selected.witness_role is not None,
        )
        registry_snapshot_uri = proposer.registry_snapshot_uri
        resolved_snapshot_uris = {
            proposer.registry_snapshot_uri,
            refiner.registry_snapshot_uri,
            evaluator.registry_snapshot_uri,
        }
        if selected.witness_role is not None:
            resolved_snapshot_uris.add(witness_oracle.registry_snapshot_uri)
        if resolved_snapshot_uris != {registry_snapshot_uri}:
            raise SearchError("resolved capabilities use different registry snapshots")
        environment_digest = _environment_digest()
        experiment_uri = new_experiment_uri()
        now = _now()
        snapshot = SearchExperimentSnapshot(
            experiment_uri=experiment_uri,
            state=ExperimentState.PENDING,
            request=selected,
            input=InputValidation(status=InputStatus.ACCEPTED),
            created_at=now,
            updated_at=now,
            request_digest=request_digest,
            effective_budget=effective_budget,
            registry_snapshot_uri=registry_snapshot_uri,
            proposer_digest=proposer.implementation_digest,
            refiner_digest=refiner.implementation_digest,
            evaluator_digest=evaluator.implementation_digest,
            environment_digest=environment_digest,
        )

        created = False
        with self.store.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing_handle = self._reuse_request(
                connection,
                selected,
                request_digest,
            )
            if existing_handle is not None:
                return existing_handle
            connection.execute(
                """
                INSERT INTO search_experiments (
                    experiment_uri, state, snapshot_json
                ) VALUES (?, ?, ?)
                """,
                (
                    experiment_uri,
                    snapshot.state.value,
                    canonicalize_json(snapshot.model_dump(mode="json")),
                ),
            )
            connection.execute(
                """
                INSERT INTO search_idempotency (
                    idempotency_key, request_digest, experiment_uri
                ) VALUES (?, ?, ?)
                """,
                (
                    selected.idempotency_key,
                    request_digest,
                    experiment_uri,
                ),
            )
            self._append_event(
                connection,
                experiment_uri,
                event_type="REQUEST_ACCEPTED",
                payload={
                    "request": selected.model_dump(mode="json"),
                    "request_digest": request_digest,
                    "effective_budget": effective_budget.model_dump(mode="json"),
                    "plugin_identity": {
                        "plugin_id": selected.plugin_id,
                        "registry_snapshot_uri": registry_snapshot_uri,
                        "proposer_digest": proposer.implementation_digest,
                        "refiner_digest": refiner.implementation_digest,
                        "evaluator_digest": evaluator.implementation_digest,
                    },
                    "environment_digest": environment_digest,
                },
            )
            created = True

        if created:
            self._launch(experiment_uri, lifecycle_reserved=True)
        return ExperimentHandle(
            experiment_uri=experiment_uri,
            state=ExperimentState.PENDING,
        )

    def _reuse_request(
        self,
        connection: sqlite3.Connection,
        request: SearchRunRequest,
        request_digest: str,
    ) -> ExperimentHandle | None:
        existing = connection.execute(
            """
            SELECT request_digest, experiment_uri
            FROM search_idempotency
            WHERE idempotency_key = ?
            """,
            (request.idempotency_key,),
        ).fetchone()
        if existing is None:
            return None
        if existing["request_digest"] != request_digest:
            raise SearchError(
                "This idempotency key is already bound to a different request. "
                "Reuse the original request or choose a new idempotency key."
            )
        existing_snapshot = self._read_snapshot(
            connection,
            existing["experiment_uri"],
        )
        self._append_event(
            connection,
            existing_snapshot.experiment_uri,
            event_type="REQUEST_REUSED",
            payload={
                "idempotency_key": request.idempotency_key,
                "request_digest": request_digest,
                "accepted_experiment_uri": existing_snapshot.experiment_uri,
            },
        )
        return ExperimentHandle(
            experiment_uri=existing_snapshot.experiment_uri,
            state=existing_snapshot.state,
        )

    def inspect(self, experiment_uri: str) -> SearchExperimentSnapshot:
        """Read the latest durable search snapshot."""

        with self.store.connection() as connection:
            return self._read_snapshot(connection, experiment_uri)

    def contains(self, experiment_uri: str) -> bool:
        """Return whether this service owns the experiment identity."""

        with self.store.connection() as connection:
            row = connection.execute(
                """
                SELECT 1
                FROM search_experiments
                WHERE experiment_uri = ?
                """,
                (experiment_uri,),
            ).fetchone()
        return row is not None

    def wait(
        self,
        experiment_uri: str,
        *,
        timeout_seconds: float = 30,
    ) -> SearchExperimentSnapshot:
        """Wait until the search is paused or terminal."""

        return wait_for_worker_settlement(
            lock=self._thread_lock,
            workers=self._threads,
            worker_id=experiment_uri,
            inspect=self.inspect,
            is_settled=lambda snapshot: snapshot.state in _SETTLED_STATES,
            timeout_seconds=timeout_seconds,
            timeout_message=(
                "The search is still running. Inspect the experiment or wait again "
                "with a larger timeout."
            ),
        )

    def pause(self, experiment_uri: str) -> ExperimentControlResult:
        """Request a pause at the next committed checkpoint boundary."""

        return self._request_control(
            experiment_uri,
            requested_state=ExperimentState.PAUSE_REQUESTED,
            event_type="PAUSE_REQUESTED",
            detail="pause requested",
        )

    def cancel(self, experiment_uri: str) -> ExperimentControlResult:
        """Request cancellation without deleting committed lineage."""

        result = self._request_control(
            experiment_uri,
            requested_state=ExperimentState.CANCEL_REQUESTED,
            event_type="CANCEL_REQUESTED",
            detail="cancellation requested",
        )
        if result.accepted and result.state == ExperimentState.CANCEL_REQUESTED:
            self._launch(experiment_uri)
        return result

    def resume(self, experiment_uri: str) -> ExperimentControlResult:
        """Resume the same invocation from its immutable checkpoint."""

        self._require_open()
        with self.store.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            snapshot = self._read_snapshot(connection, experiment_uri)
            if snapshot.state != ExperimentState.PAUSED:
                return ExperimentControlResult(
                    experiment_uri=experiment_uri,
                    state=snapshot.state,
                    accepted=False,
                    detail="only a paused search can resume",
                )
            resumed = _updated_snapshot(
                snapshot,
                state=ExperimentState.PENDING,
                updated_at=_now(),
                detail="resume requested from committed checkpoint",
            )
            self._update_snapshot(connection, resumed)
            self._append_event(
                connection,
                experiment_uri,
                event_type="RESUME_REQUESTED",
                payload={
                    "checkpoint_uri": resumed.checkpoint_uri,
                    "accounting": resumed.accounting.model_dump(mode="json"),
                },
            )
        self._launch(experiment_uri)
        return ExperimentControlResult(
            experiment_uri=experiment_uri,
            state=ExperimentState.PENDING,
            accepted=True,
            detail="resume requested",
        )

    def events(self, experiment_uri: str) -> tuple[SearchLifecycleEvent, ...]:
        """Return the validated append-only lifecycle event chain."""

        with self.store.connection() as connection:
            if (
                connection.execute(
                    """
                    SELECT 1 FROM search_experiments
                    WHERE experiment_uri = ?
                    """,
                    (experiment_uri,),
                ).fetchone()
                is None
            ):
                _LOGGER.warning("search experiment not found: %s", experiment_uri)
                raise SearchError(
                    "The experiment was not found. Check the URI returned by "
                    "search.run or search.enumerate, or start a new experiment."
                )
            rows = connection.execute(
                """
                SELECT sequence, event_json
                FROM search_events
                WHERE experiment_uri = ?
                ORDER BY sequence
                """,
                (experiment_uri,),
            ).fetchall()
        try:
            events = tuple(
                decode_persisted_model(
                    SearchLifecycleEvent,
                    row["event_json"],
                    record_kind="search_event",
                    record_id=f"{experiment_uri}:{row['sequence']}",
                    field="event_json",
                )
                for row in rows
            )
        except PersistenceCorruptionError as exc:
            raise SearchCorruptionError(exc) from exc
        previous: str | None = None
        for event in events:
            if event.previous_event_digest != previous:
                raise SearchError("stored search event chain is invalid")
            expected = _event_digest(event)
            if event.event_digest != expected:
                raise SearchError("stored search event digest is invalid")
            previous = event.event_digest
        return events

    def _effective_budget(
        self,
        requested: SearchBudget,
        *,
        include_witness_lineage: bool,
    ) -> SearchBudget:
        """Apply the restrictive intersection of request and operator limits.

        Witness-enabled search requires more parent slots than ordinary search
        because each candidate lineage page records three lineage parents (one
        per witness parent) instead of one. The minimum ``max_parents`` is
        ``fixed_page_parents + parents_per_candidate``: 4 for ordinary search,
        6 for witness-enabled search.
        """

        fixed_page_parents = 3  # claim, plugin, and one shared evaluation
        parents_per_candidate = 3 if include_witness_lineage else 1
        minimum_parent_capacity = fixed_page_parents + parents_per_candidate
        lineage_batch_size = (
            self.store.limits.max_parents - fixed_page_parents
        ) // parents_per_candidate
        if lineage_batch_size < 1:
            search_kind = (
                "witness-enabled search" if include_witness_lineage else "search"
            )
            raise SearchError(
                "store parent limit must be at least "
                f"{minimum_parent_capacity} for one {search_kind} archive record"
            )
        return SearchBudget(
            candidates_max=min(requested.candidates_max, self.max_candidates),
            iterations_max=min(requested.iterations_max, self.max_iterations),
            wall_seconds=min(requested.wall_seconds, self.max_wall_seconds),
            batch_size=min(
                requested.batch_size,
                self.max_batch_size,
                self.evaluation.max_batch_size,
                lineage_batch_size,
            ),
            workers=requested.workers,
        )

    def _launch(
        self,
        experiment_uri: str,
        *,
        lifecycle_reserved: bool = False,
    ) -> None:
        status = launch_worker(
            lock=self._thread_lock,
            lifecycle_state=lambda: self._lifecycle_state,
            workers=self._threads,
            worker_id=experiment_uri,
            target=self._run,
            name=f"jacobian-search-{experiment_uri.removeprefix('experiment://')}",
            lifecycle_reserved=lifecycle_reserved,
        )
        if status is WorkerLaunchStatus.SERVICE_CLOSING:
            raise SearchError("search service is closing")

    def _budget_exhausted(
        self,
        experiment_uri: str,
        accounting: SearchAccounting,
        budget: SearchBudget,
        *,
        wall_time_ms: int,
    ) -> bool:
        """Check if the search budget is exhausted, finishing if so.

        Returns True when the search has been finished as TIMEOUT or COMPLETED.
        The caller should return immediately when this returns True.
        """

        if wall_time_ms >= budget.wall_seconds * 1000:
            self._finish(
                experiment_uri,
                state=ExperimentState.TIMEOUT,
                stop_reason=SearchStopReason.WALL_TIME_LIMIT,
                strategy_complete=False,
                detail="search wall-clock budget exhausted",
                wall_time_ms=wall_time_ms,
            )
            return True
        if accounting.iterations >= budget.iterations_max:
            self._finish(
                experiment_uri,
                state=ExperimentState.COMPLETED,
                stop_reason=SearchStopReason.ITERATION_LIMIT,
                strategy_complete=False,
                detail="search iteration limit reached",
                wall_time_ms=wall_time_ms,
            )
            return True
        if accounting.proposed_candidates >= budget.candidates_max:
            self._finish(
                experiment_uri,
                state=ExperimentState.COMPLETED,
                stop_reason=SearchStopReason.CANDIDATE_LIMIT,
                strategy_complete=False,
                detail="search candidate limit reached",
                wall_time_ms=wall_time_ms,
            )
            return True
        return False

    def _run(self, experiment_uri: str) -> None:
        from jacobian.search.run_loop import execute_search

        execute_search(self, experiment_uri)

    def _resolve_strategy(
        self,
        snapshot: SearchExperimentSnapshot,
    ) -> tuple[ResolvedCapability, ResolvedCapability, str]:
        if (
            snapshot.proposer_digest is None
            or snapshot.refiner_digest is None
            or snapshot.evaluator_digest is None
        ):
            raise SearchError("search snapshot is missing implementation identity")
        proposer = self.plugins.resolve(
            snapshot.request.plugin_id,
            CapabilityName.PROPOSER,
        )
        refiner = self.plugins.resolve(
            snapshot.request.plugin_id,
            CapabilityName.REFINER,
        )
        evaluator = self.plugins.resolve(
            snapshot.request.plugin_id,
            CapabilityName.EVALUATOR,
        )
        if {
            proposer.registry_snapshot_uri,
            refiner.registry_snapshot_uri,
            evaluator.registry_snapshot_uri,
        } != {snapshot.registry_snapshot_uri}:
            raise SearchError("resolved strategy uses a different registry snapshot")
        if proposer.implementation_digest != snapshot.proposer_digest:
            raise SearchError("proposer identity changed after request acceptance")
        if refiner.implementation_digest != snapshot.refiner_digest:
            raise SearchError("refiner identity changed after request acceptance")
        if evaluator.implementation_digest != snapshot.evaluator_digest:
            raise SearchError("evaluator identity changed after request acceptance")
        return proposer, refiner, evaluator.implementation_digest

    def _mark_running(
        self,
        snapshot: SearchExperimentSnapshot,
    ) -> ExperimentState:
        with self.store.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._read_snapshot(connection, snapshot.experiment_uri)
            if current.state == ExperimentState.PAUSE_REQUESTED:
                paused = _updated_snapshot(
                    current,
                    state=ExperimentState.PAUSED,
                    updated_at=_now(),
                    detail="search paused before new work began",
                )
                self._update_snapshot(connection, paused)
                self._append_event(
                    connection,
                    current.experiment_uri,
                    event_type="PAUSED",
                    payload={"checkpoint_uri": paused.checkpoint_uri},
                )
                return ExperimentState.PAUSED
            if current.state == ExperimentState.CANCEL_REQUESTED:
                return ExperimentState.CANCEL_REQUESTED
            if current.state != ExperimentState.PENDING:
                raise SearchError(f"cannot start search from {current.state.value}")
            running = _updated_snapshot(
                current,
                state=ExperimentState.RUNNING,
                updated_at=_now(),
                detail="search running",
            )
            self._update_snapshot(connection, running)
            self._append_event(
                connection,
                current.experiment_uri,
                event_type="RUNNING",
                payload={"checkpoint_uri": running.checkpoint_uri},
            )
        return ExperimentState.RUNNING

    def _commit_progress(
        self,
        progress: SearchExperimentSnapshot,
    ) -> ExperimentState:
        with self.store.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._read_snapshot(connection, progress.experiment_uri)
            if current.state == ExperimentState.PAUSE_REQUESTED:
                committed = _updated_snapshot(
                    progress,
                    state=ExperimentState.PAUSED,
                    detail="search paused at a committed checkpoint",
                )
                event_type = "PAUSED"
            elif current.state == ExperimentState.CANCEL_REQUESTED:
                committed = _updated_snapshot(
                    progress,
                    state=ExperimentState.CANCEL_REQUESTED,
                    detail="checkpoint committed before cancellation",
                )
                event_type = "CHECKPOINT_COMMITTED"
            elif current.state == ExperimentState.RUNNING:
                committed = progress
                event_type = "CHECKPOINT_COMMITTED"
            else:
                raise SearchError(f"cannot commit progress from {current.state.value}")
            self._update_snapshot(connection, committed)
            self._append_event(
                connection,
                committed.experiment_uri,
                event_type=event_type,
                payload={
                    "checkpoint_uri": committed.checkpoint_uri,
                    "archive_page_uri": committed.archive_page_uris[-1],
                    "accounting": committed.accounting.model_dump(mode="json"),
                },
            )
        return committed.state

    def _finish_execution_failure(
        self,
        experiment_uri: str,
        execution_status: ExecutionStatus,
        detail: str,
        *,
        wall_time_ms: int,
        accounting_override: SearchAccounting | None = None,
    ) -> None:
        if execution_status == ExecutionStatus.TIMEOUT:
            self._finish(
                experiment_uri,
                state=ExperimentState.TIMEOUT,
                stop_reason=SearchStopReason.WALL_TIME_LIMIT,
                strategy_complete=False,
                detail=detail,
                wall_time_ms=wall_time_ms,
                accounting_override=accounting_override,
            )
            return
        self._finish(
            experiment_uri,
            state=ExperimentState.ERROR,
            stop_reason=SearchStopReason.ERROR,
            strategy_complete=False,
            detail=detail,
            wall_time_ms=wall_time_ms,
            accounting_override=accounting_override,
        )

    def _store_archive(
        self,
        snapshot: SearchExperimentSnapshot,
        accounting: SearchAccounting,
    ) -> ArtifactPutResult:
        manifest = SearchArchiveManifest(
            experiment_uri=snapshot.experiment_uri,
            request_digest=snapshot.request_digest,
            claim_uri=snapshot.request.claim_uri,
            plugin_id=snapshot.request.plugin_id,
            registry_snapshot_uri=snapshot.registry_snapshot_uri,
            page_uris=snapshot.archive_page_uris,
            accounting=accounting,
            effective_budget=snapshot.effective_budget,
            environment_digest=snapshot.environment_digest,
        )
        return self._put_internal_artifact(
            schema_uri=self.archive_manifest_schema_uri,
            payload=manifest.model_dump(mode="json"),
            parents=(
                snapshot.request.claim_uri,
                snapshot.request.plugin_id,
                *((snapshot.checkpoint_uri,) if snapshot.checkpoint_uri else ()),
            ),
            summary="search archive manifest",
        )

    def _finish(
        self,
        experiment_uri: str,
        *,
        state: ExperimentState,
        stop_reason: SearchStopReason,
        strategy_complete: bool,
        detail: str,
        wall_time_ms: int,
        accounting_override: SearchAccounting | None = None,
    ) -> None:
        current = self.inspect(experiment_uri)
        terminal_accounting = _updated_accounting(
            accounting_override or current.accounting,
            wall_time_ms=max(current.accounting.wall_time_ms, wall_time_ms),
        )
        archive = self._store_archive(current, terminal_accounting)
        terminal = _updated_snapshot(
            current,
            state=state,
            updated_at=_now(),
            stop_reason=stop_reason,
            strategy_reported_complete=strategy_complete,
            verification=Verification.UNVERIFIED,
            archive_uri=archive.artifact_uri,
            accounting=terminal_accounting,
            detail=detail,
        )
        with self.store.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            latest = self._read_snapshot(connection, experiment_uri)
            if latest.state == ExperimentState.CANCEL_REQUESTED:
                terminal = _updated_snapshot(
                    terminal,
                    state=ExperimentState.CANCELLED,
                    stop_reason=SearchStopReason.CANCELLED,
                    strategy_reported_complete=False,
                    detail=f"search cancelled; {detail}",
                )
            elif latest.state in _TERMINAL_STATES:
                raise SearchError(f"search is already terminal: {latest.state.value}")
            self._update_snapshot(connection, terminal)
            self._append_event(
                connection,
                experiment_uri,
                event_type=terminal.state.value,
                payload={
                    "stop_reason": terminal.stop_reason,
                    "archive_uri": terminal.archive_uri,
                    "checkpoint_uri": terminal.checkpoint_uri,
                    "accounting": terminal.accounting.model_dump(mode="json"),
                    "detail": terminal.detail,
                },
            )

    def _finish_if_possible(
        self,
        experiment_uri: str,
        *,
        state: ExperimentState,
        stop_reason: SearchStopReason,
        detail: str,
        wall_time_ms: int,
        accounting_override: SearchAccounting | None = None,
    ) -> None:
        try:
            snapshot = self.inspect(experiment_uri)
            if snapshot.state in _TERMINAL_STATES:
                return
            self._finish(
                experiment_uri,
                state=state,
                stop_reason=stop_reason,
                strategy_complete=False,
                detail=detail,
                wall_time_ms=wall_time_ms,
                accounting_override=accounting_override,
            )
        except (
            SearchError,
            StorageError,
            SchemaRegistryError,
            ValidationError,
        ) as exc:
            self._record_terminal_persistence_failure(
                experiment_uri,
                detail=detail,
                error=exc,
                wall_time_ms=wall_time_ms,
                accounting_override=accounting_override,
            )

    def _record_terminal_persistence_failure(
        self,
        experiment_uri: str,
        *,
        detail: str,
        error: Exception,
        wall_time_ms: int,
        accounting_override: SearchAccounting | None,
    ) -> None:
        """Fail closed when the terminal archive cannot be committed."""

        _LOGGER.warning(
            "search terminal archive persistence failed",
            exc_info=error,
        )
        with self.store.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._read_snapshot(connection, experiment_uri)
            if current.state in _TERMINAL_STATES:
                return
            terminal_detail = (
                "Jacobian could not save the final experiment archive. "
                "The experiment remains unverified. Check state-directory space and "
                "permissions, then inspect the experiment."
            )
            terminal = _updated_snapshot(
                current,
                state=ExperimentState.ERROR,
                updated_at=_now(),
                stop_reason=SearchStopReason.ERROR,
                strategy_reported_complete=False,
                verification=Verification.UNVERIFIED,
                archive_uri=None,
                accounting=_updated_accounting(
                    accounting_override or current.accounting,
                    wall_time_ms=max(
                        current.accounting.wall_time_ms,
                        wall_time_ms,
                    ),
                ),
                detail=terminal_detail,
            )
            self._update_snapshot(connection, terminal)
            self._append_event(
                connection,
                experiment_uri,
                event_type=ExperimentState.ERROR.value,
                payload={
                    "stop_reason": terminal.stop_reason,
                    "archive_uri": None,
                    "checkpoint_uri": terminal.checkpoint_uri,
                    "accounting": terminal.accounting.model_dump(mode="json"),
                    "detail": terminal.detail,
                },
            )

    def _request_control(
        self,
        experiment_uri: str,
        *,
        requested_state: ExperimentState,
        event_type: str,
        detail: str,
    ) -> ExperimentControlResult:
        with self.store.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            snapshot = self._read_snapshot(connection, experiment_uri)
            if snapshot.state in _TERMINAL_STATES:
                return ExperimentControlResult(
                    experiment_uri=experiment_uri,
                    state=snapshot.state,
                    accepted=False,
                    detail="search is already terminal",
                )
            if requested_state == ExperimentState.PAUSE_REQUESTED and (
                snapshot.state
                in {ExperimentState.PAUSED, ExperimentState.CANCEL_REQUESTED}
            ):
                return ExperimentControlResult(
                    experiment_uri=experiment_uri,
                    state=snapshot.state,
                    accepted=False,
                    detail="search cannot pause from its current state",
                )
            if requested_state == ExperimentState.CANCEL_REQUESTED and (
                snapshot.state == ExperimentState.CANCEL_REQUESTED
            ):
                return ExperimentControlResult(
                    experiment_uri=experiment_uri,
                    state=snapshot.state,
                    accepted=False,
                    detail="cancellation is already requested",
                )
            controlled = _updated_snapshot(
                snapshot,
                state=requested_state,
                updated_at=_now(),
                detail=detail,
            )
            self._update_snapshot(connection, controlled)
            self._append_event(
                connection,
                experiment_uri,
                event_type=event_type,
                payload={"checkpoint_uri": controlled.checkpoint_uri},
            )
        return ExperimentControlResult(
            experiment_uri=experiment_uri,
            state=requested_state,
            accepted=True,
            detail=detail,
        )

    def _record_operation(
        self,
        experiment_uri: str,
        *,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        with self.store.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._append_event(
                connection,
                experiment_uri,
                event_type=event_type,
                payload=payload,
            )

    def _put_internal_artifact(
        self,
        *,
        schema_uri: str,
        payload: Any,
        parents: tuple[str, ...] = (),
        summary: str,
    ) -> ArtifactPutResult:
        return put_internal_artifact(
            self.store,
            self.schemas,
            self.semantics_uri,
            schema_uri=schema_uri,
            payload=payload,
            parents=parents,
            summary=summary,
        )

    def _read_snapshot(
        self,
        connection: sqlite3.Connection,
        experiment_uri: str,
    ) -> SearchExperimentSnapshot:
        row = connection.execute(
            """
            SELECT snapshot_json
            FROM search_experiments
            WHERE experiment_uri = ?
            """,
            (experiment_uri,),
        ).fetchone()
        if row is None:
            _LOGGER.warning("search experiment not found: %s", experiment_uri)
            raise SearchError(
                "The experiment was not found. Check the URI returned by search.run "
                "or search.enumerate, or start a new experiment."
            )
        try:
            return decode_persisted_model(
                SearchExperimentSnapshot,
                row["snapshot_json"],
                record_kind="search_snapshot",
                record_id=experiment_uri,
                field="snapshot_json",
            )
        except PersistenceCorruptionError as exc:
            raise SearchCorruptionError(exc) from exc
        except (ValidationError, ValueError) as exc:
            raise SearchError("stored search snapshot is invalid") from exc

    @staticmethod
    def _update_snapshot(
        connection: sqlite3.Connection,
        snapshot: SearchExperimentSnapshot,
    ) -> None:
        connection.execute(
            """
            UPDATE search_experiments
            SET state = ?, snapshot_json = ?
            WHERE experiment_uri = ?
            """,
            (
                snapshot.state.value,
                canonicalize_json(snapshot.model_dump(mode="json")),
                snapshot.experiment_uri,
            ),
        )

    @staticmethod
    def _append_event(
        connection: sqlite3.Connection,
        experiment_uri: str,
        *,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        row = connection.execute(
            """
            SELECT sequence, event_digest
            FROM search_events
            WHERE experiment_uri = ?
            ORDER BY sequence DESC
            LIMIT 1
            """,
            (experiment_uri,),
        ).fetchone()
        sequence = 0 if row is None else int(row["sequence"]) + 1
        previous_digest = None if row is None else str(row["event_digest"])
        occurred_at = _now()
        unsigned_event = SearchLifecycleEvent(
            experiment_uri=experiment_uri,
            sequence=sequence,
            event_type=event_type,
            occurred_at=occurred_at,
            payload=payload,
            previous_event_digest=previous_digest,
            event_digest="sha256:" + "0" * 64,
        )
        event = unsigned_event.model_copy(
            update={"event_digest": _event_digest(unsigned_event)}
        )
        connection.execute(
            """
            INSERT INTO search_events (
                experiment_uri, sequence, event_json, event_digest
            ) VALUES (?, ?, ?, ?)
            """,
            (
                experiment_uri,
                sequence,
                canonicalize_json(event.model_dump(mode="json")),
                event.event_digest,
            ),
        )
