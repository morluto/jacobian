"""Persistent bounded-enumeration experiments."""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from jacobian.canonical import canonicalize_json
from jacobian.claims import ClaimValidationService
from jacobian.contracts.artifacts import ArtifactPutResult
from jacobian.contracts.discovery import (
    EnumerationAccounting,
    EnumerationArchive,
    EnumerationArchiveManifest,
    EnumerationStopReason,
    ExperimentCancelResult,
    ExperimentHandle,
    ExperimentSnapshot,
    ExperimentState,
    PluginEnumerationPage,
    SearchEnumerateRequest,
)
from jacobian.contracts.evaluation import EvaluationBatchResult
from jacobian.contracts.plugins import CapabilityName
from jacobian.contracts.results import (
    Coverage,
    ExecutionStatus,
    InputStatus,
    InputValidation,
    Verification,
)
from jacobian.evaluation import (
    EvaluationService,
    require_complete_evaluation_batch,
)
from jacobian.experiment_identity import new_experiment_uri
from jacobian.experiments._helpers import (
    _enumeration_failure_detail,
    _now,
    _updated,
)
from jacobian.experiments.errors import (
    ExperimentCorruptionError,
    ExperimentError,
    ExperimentNotFoundError,
)
from jacobian.lifecycle import (
    LifecycleTimeoutError,
    ServiceLifecycleState,
    WorkerLaunchStatus,
    launch_worker,
    wait_for_worker_quiescence,
    wait_for_worker_settlement,
)
from jacobian.persistence import PersistenceCorruptionError, decode_persisted_model
from jacobian.plugin_execution import PluginExecutor
from jacobian.plugins.registry import PluginRegistry, PluginRegistryError
from jacobian.schema_registry import SchemaRegistry, SchemaRegistryError, model_schema
from jacobian.storage.errors import StorageError
from jacobian.storage.repository import ArtifactRepository
from jacobian.structures import StructureService

_LOGGER = logging.getLogger(__name__)

_TERMINAL_STATES = {
    ExperimentState.COMPLETED,
    ExperimentState.CANCELLED,
    ExperimentState.TIMEOUT,
    ExperimentState.ERROR,
}
_EXPERIMENT_NOT_FOUND = (
    "The experiment was not found. Check the URI returned by search.run or "
    "search.enumerate, or start a new experiment."
)


def _decode_experiment_snapshot(
    encoded: str | bytes | bytearray,
    experiment_uri: str,
) -> ExperimentSnapshot:
    try:
        return decode_persisted_model(
            ExperimentSnapshot,
            encoded,
            record_kind="experiment_snapshot",
            record_id=experiment_uri,
            field="snapshot_json",
        )
    except PersistenceCorruptionError as exc:
        raise ExperimentCorruptionError(exc) from exc


@dataclass
class _EnumerationRunState:
    page_uris: list[str] = field(default_factory=list)
    seen_keys: set[str] = field(default_factory=set)
    raw_candidates: int = 0
    unique_candidates: int = 0
    duplicate_candidates: int = 0
    evaluated_candidates: int = 0
    pages: int = 0
    cursor: dict[str, Any] | None = None
    scope_bytes: bytes | None = None
    scope_uri: str | None = None

    def accounting(self) -> EnumerationAccounting:
        return EnumerationAccounting(
            raw_candidates=self.raw_candidates,
            unique_candidates=self.unique_candidates,
            duplicate_candidates=self.duplicate_candidates,
            evaluated_candidates=self.evaluated_candidates,
            pages=self.pages,
        )


class ExperimentService:
    """Run local enumeration jobs while persisting auditable progress."""

    def __init__(
        self,
        store: ArtifactRepository,
        schemas: SchemaRegistry,
        plugins: PluginRegistry,
        claims: ClaimValidationService,
        executor: PluginExecutor,
        evaluation: EvaluationService,
        structures: StructureService,
    ) -> None:
        self.store = store
        self.schemas = schemas
        self.plugins = plugins
        self.claims = claims
        self.executor = executor
        self.evaluation = evaluation
        self.structures = structures
        self._threads: dict[str, threading.Thread] = {}
        self._thread_lock = threading.Lock()
        self._starts_in_flight = 0
        self._lifecycle_state = ServiceLifecycleState.OPEN
        self._recover_interrupted_experiments()
        self.semantics_uri = store.register_descriptor(
            kind="semantics",
            name="jacobian.bounded-enumeration",
            version="1",
            definition={
                "description": (
                    "untrusted bounded enumeration scope, progress, and archive"
                )
            },
        )
        self.scope_schema_uri = schemas.register(
            name="jacobian.enumeration-scope",
            version="1",
            schema={"type": "object"},
        )
        self.archive_page_schema_uri = schemas.register(
            name="jacobian.enumeration-archive-page",
            version="1",
            schema=model_schema(EnumerationArchive),
        )
        self.archive_manifest_schema_uri = schemas.register(
            name="jacobian.enumeration-archive",
            version="1",
            schema=model_schema(EnumerationArchiveManifest),
        )
        self.evaluation_schema_uri = schemas.register(
            name="jacobian.evaluation-batch-result",
            version="1",
            schema=model_schema(EvaluationBatchResult),
        )

    def close(self, *, timeout_seconds: float = 30) -> None:
        """Quiesce enumeration starts and workers before storage teardown."""

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
            raise ExperimentError(
                "enumeration workers did not quiesce before runtime shutdown"
            ) from exc
        with self._thread_lock:
            self._lifecycle_state = ServiceLifecycleState.CLOSED

    def _put_internal_artifact(
        self,
        *,
        schema_uri: str,
        payload: Any,
        parents: tuple[str, ...] = (),
        summary: str,
    ) -> ArtifactPutResult:
        """Validate runtime-owned experiment data before committing it."""

        normalized = self.schemas.validate(schema_uri, payload)
        self.store.get_descriptor(
            self.semantics_uri,
            expected_kind="semantics",
        )
        return self.store.put(
            schema_uri=schema_uri,
            semantics_uri=self.semantics_uri,
            payload=normalized,
            parents=parents,
            summary=summary,
        )

    def _recover_interrupted_experiments(self) -> None:
        with self.store.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """
                SELECT experiment_uri, state, snapshot_json
                FROM experiments
                WHERE state IN ('PENDING', 'RUNNING', 'CANCEL_REQUESTED')
                OR state NOT IN (
                    'PENDING', 'RUNNING', 'COMPLETED',
                    'CANCEL_REQUESTED', 'CANCELLED', 'TIMEOUT', 'ERROR'
                )
                """
            ).fetchall()
            for row in rows:
                try:
                    snapshot = _decode_experiment_snapshot(
                        row["snapshot_json"], str(row["experiment_uri"])
                    )
                except (
                    TypeError,
                    ValidationError,
                    ValueError,
                    ExperimentCorruptionError,
                ) as exc:
                    self._quarantine_recovery_snapshot(connection, row, exc)
                    continue
                if snapshot.experiment_uri != str(
                    row["experiment_uri"]
                ) or snapshot.state.value != str(row["state"]):
                    self._quarantine_recovery_snapshot(
                        connection,
                        row,
                        ValueError(
                            "stored enumeration snapshot identity or state differs "
                            "from its database index"
                        ),
                    )
                    continue
                interrupted = _updated(
                    snapshot,
                    state=ExperimentState.ERROR,
                    stop_reason=EnumerationStopReason.ERROR,
                    updated_at=_now(),
                    detail="experiment process ended before completion",
                )
                connection.execute(
                    """
                    UPDATE experiments
                    SET state = ?, snapshot_json = ?
                    WHERE experiment_uri = ?
                    """,
                    (
                        interrupted.state.value,
                        canonicalize_json(interrupted.model_dump(mode="json")),
                        row["experiment_uri"],
                    ),
                )

    @staticmethod
    def _quarantine_recovery_snapshot(
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        error: Exception,
    ) -> None:
        """Isolate one invalid recovery row without blocking valid experiments."""

        experiment_uri = str(row["experiment_uri"])
        raw = row["snapshot_json"]
        if isinstance(raw, bytes):
            raw_bytes = raw
        elif isinstance(raw, str):
            raw_bytes = raw.encode("utf-8")
        else:
            raw_bytes = repr(raw).encode("utf-8")
        snapshot_digest = "sha256:" + hashlib.sha256(raw_bytes).hexdigest()
        detail = (
            "Stored experiment state is invalid. Restore the Jacobian state "
            "directory from a trusted backup or start a new experiment."
        )
        _LOGGER.warning(
            "quarantining invalid enumeration snapshot for %s",
            experiment_uri,
            exc_info=error,
        )
        connection.execute(
            """
            UPDATE experiments
            SET state = 'ERROR'
            WHERE experiment_uri = ?
            """,
            (experiment_uri,),
        )
        connection.execute(
            """
            INSERT OR REPLACE INTO experiment_recovery_failures (
                experiment_uri, detected_at, snapshot_digest, detail
            ) VALUES (?, ?, ?, ?)
            """,
            (
                experiment_uri,
                _now().isoformat(),
                snapshot_digest,
                detail,
            ),
        )

    def start_enumeration(
        self,
        request: SearchEnumerateRequest | dict[str, Any],
    ) -> ExperimentHandle:
        """Validate, persist, and launch one bounded enumeration job."""

        with self._thread_lock:
            if self._lifecycle_state is not ServiceLifecycleState.OPEN:
                raise ExperimentError("experiment service is closing")
            self._starts_in_flight += 1
        try:
            return self._start_enumeration_reserved(request)
        finally:
            with self._thread_lock:
                self._starts_in_flight -= 1

    def _start_enumeration_reserved(
        self,
        request: SearchEnumerateRequest | dict[str, Any],
    ) -> ExperimentHandle:
        """Start after reserving the lifecycle through worker launch."""

        selected = SearchEnumerateRequest.model_validate(request)
        validation = self.claims.validate(
            claim_uri=selected.claim_uri,
            plugin_id=selected.plugin_id,
        )
        if not validation.valid:
            raise ExperimentError("; ".join(validation.input.errors))
        try:
            self.plugins.resolve(
                selected.plugin_id,
                CapabilityName.CANDIDATE_ENUMERATOR,
            )
            self.plugins.resolve(selected.plugin_id, CapabilityName.EVALUATOR)
            if selected.quotient_by_isomorphism:
                self.plugins.resolve(
                    selected.plugin_id,
                    CapabilityName.CANONICALIZER,
                )
        except PluginRegistryError as exc:
            _LOGGER.warning("enumeration capability resolution failed", exc_info=exc)
            capability_hint = (
                "enumeration, evaluation, and Canonicalizer capabilities"
                if selected.quotient_by_isomorphism
                else "enumeration and evaluation capabilities"
            )
            raise ExperimentError(
                "The enumeration plugin is unavailable or incomplete. Call "
                "math.find, choose a reference domain with "
                f"{capability_hint}, then retry."
            ) from exc

        experiment_uri = new_experiment_uri()
        now = _now()
        snapshot = ExperimentSnapshot(
            experiment_uri=experiment_uri,
            state=ExperimentState.PENDING,
            request=selected,
            input=InputValidation(status=InputStatus.ACCEPTED),
            created_at=now,
            updated_at=now,
        )
        self._write_new(snapshot)
        self._launch_enumeration(experiment_uri, lifecycle_reserved=True)
        return ExperimentHandle(
            experiment_uri=experiment_uri,
            state=ExperimentState.PENDING,
        )

    def _launch_enumeration(
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
            target=self._run_enumeration,
            name=(
                f"jacobian-enumeration-{experiment_uri.removeprefix('experiment://')}"
            ),
            lifecycle_reserved=lifecycle_reserved,
        )
        if status is WorkerLaunchStatus.SERVICE_CLOSING:
            raise ExperimentError("experiment service is closing")

    def inspect(self, experiment_uri: str) -> ExperimentSnapshot:
        """Read the latest durable experiment snapshot."""

        with self.store.connection() as connection:
            row = connection.execute(
                """
                SELECT snapshot_json
                FROM experiments
                WHERE experiment_uri = ?
                """,
                (experiment_uri,),
            ).fetchone()
        if row is None:
            _LOGGER.warning("experiment not found: %s", experiment_uri)
            raise ExperimentNotFoundError(_EXPERIMENT_NOT_FOUND)
        try:
            return _decode_experiment_snapshot(row["snapshot_json"], experiment_uri)
        except (ValidationError, ValueError, ExperimentCorruptionError) as exc:
            raise ExperimentError("stored experiment snapshot is invalid") from exc

    def contains(self, experiment_uri: str) -> bool:
        """Return whether this service owns the experiment identity."""

        with self.store.connection() as connection:
            row = connection.execute(
                """
                SELECT 1
                FROM experiments
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
    ) -> ExperimentSnapshot:
        """Wait locally for a terminal state, then return its snapshot."""

        return wait_for_worker_settlement(
            lock=self._thread_lock,
            workers=self._threads,
            worker_id=experiment_uri,
            inspect=self.inspect,
            is_settled=lambda snapshot: snapshot.state in _TERMINAL_STATES,
            timeout_seconds=timeout_seconds,
            timeout_message=(
                "The experiment is still running. Inspect it or wait again with a "
                "larger timeout."
            ),
        )

    def cancel(self, experiment_uri: str) -> ExperimentCancelResult:
        """Request cooperative cancellation without deleting committed artifacts."""

        with self.store.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT snapshot_json
                FROM experiments
                WHERE experiment_uri = ?
                """,
                (experiment_uri,),
            ).fetchone()
            if row is None:
                _LOGGER.warning(
                    "experiment not found: %s",
                    experiment_uri,
                )
                raise ExperimentNotFoundError(_EXPERIMENT_NOT_FOUND)
            snapshot = _decode_experiment_snapshot(row["snapshot_json"], experiment_uri)
            if snapshot.state in _TERMINAL_STATES:
                return ExperimentCancelResult(
                    experiment_uri=experiment_uri,
                    state=snapshot.state,
                    accepted=False,
                    detail="experiment is already terminal",
                )
            cancelled = _updated(
                snapshot,
                state=ExperimentState.CANCEL_REQUESTED,
                updated_at=_now(),
                detail="cancellation requested",
            )
            connection.execute(
                """
                UPDATE experiments
                SET state = ?, snapshot_json = ?
                WHERE experiment_uri = ?
                """,
                (
                    cancelled.state.value,
                    canonicalize_json(cancelled.model_dump(mode="json")),
                    experiment_uri,
                ),
            )
        return ExperimentCancelResult(
            experiment_uri=experiment_uri,
            state=ExperimentState.CANCEL_REQUESTED,
            accepted=True,
            detail="cancellation requested",
        )

    def _check_enumeration_termination(
        self,
        run_state: _EnumerationRunState,
        experiment_uri: str,
        started: float,
        request: SearchEnumerateRequest,
        remaining_wall: float,
    ) -> bool:
        """Return True when cancel, wall-time, or candidate-limit stops the run."""

        if self._cancel_requested(experiment_uri):
            self._finish(
                experiment_uri,
                state=ExperimentState.CANCELLED,
                stop_reason=EnumerationStopReason.CANCELLED,
                started=started,
                request=request,
                scope_uri=run_state.scope_uri,
                page_uris=run_state.page_uris,
                complete=False,
                accounting=run_state.accounting(),
                detail="experiment cancelled",
            )
            return True
        if remaining_wall <= 0:
            self._finish(
                experiment_uri,
                state=ExperimentState.TIMEOUT,
                stop_reason=EnumerationStopReason.WALL_TIME_LIMIT,
                started=started,
                request=request,
                scope_uri=run_state.scope_uri,
                page_uris=run_state.page_uris,
                complete=False,
                accounting=run_state.accounting(),
                detail="experiment wall-clock budget exhausted",
            )
            return True
        remaining_candidates = request.budget.candidates_max - run_state.raw_candidates
        if remaining_candidates <= 0:
            self._finish(
                experiment_uri,
                state=ExperimentState.COMPLETED,
                stop_reason=EnumerationStopReason.CANDIDATE_LIMIT,
                started=started,
                request=request,
                scope_uri=run_state.scope_uri,
                page_uris=run_state.page_uris,
                complete=False,
                accounting=run_state.accounting(),
                detail="candidate limit reached before completeness",
            )
            return True
        return False

    def _run_enumerator_page(
        self,
        run_state: _EnumerationRunState,
        experiment_uri: str,
        started: float,
        request: SearchEnumerateRequest,
        enumerator: Any,
        remaining_wall: float,
    ) -> PluginEnumerationPage | None:
        """Execute one enumerator page, validate it, and bind its scope."""

        remaining_candidates = request.budget.candidates_max - run_state.raw_candidates
        fixed_page_parents = 1 if not run_state.page_uris else 2
        page_size = min(
            request.budget.page_size,
            remaining_candidates,
            self.evaluation.max_batch_size,
            self.store.limits.max_parents - fixed_page_parents,
        )
        execution = self.executor.run(
            entrypoint=enumerator.descriptor.entrypoint,
            implementation_digest=enumerator.implementation_digest,
            request={
                "request_version": "1",
                "bounds": request.bounds,
                "cursor": run_state.cursor,
                "page_size": page_size,
                "seed": request.seed,
            },
            timeout_seconds=remaining_wall,
        )
        if execution.status != ExecutionStatus.COMPLETED:
            terminal_state = (
                ExperimentState.TIMEOUT
                if execution.status == ExecutionStatus.TIMEOUT
                else ExperimentState.ERROR
            )
            reason = (
                EnumerationStopReason.WALL_TIME_LIMIT
                if terminal_state == ExperimentState.TIMEOUT
                else EnumerationStopReason.ERROR
            )
            self._finish(
                experiment_uri,
                state=terminal_state,
                stop_reason=reason,
                started=started,
                request=request,
                scope_uri=run_state.scope_uri,
                page_uris=run_state.page_uris,
                complete=False,
                accounting=run_state.accounting(),
                detail=execution.detail or "enumerator execution failed",
            )
            return None
        page = PluginEnumerationPage.model_validate(execution.output)
        run_state.pages += 1
        encoded_scope = canonicalize_json(page.scope)
        if run_state.scope_bytes is None:
            run_state.scope_bytes = encoded_scope
            scope = self._put_internal_artifact(
                schema_uri=self.scope_schema_uri,
                payload={
                    "plugin_id": request.plugin_id,
                    "bounds": request.bounds,
                    "enumerator_scope": page.scope,
                    "enumerator_digest": enumerator.implementation_digest,
                },
                parents=(request.claim_uri, request.plugin_id),
                summary="enumeration scope reported by untrusted plugin",
            )
            run_state.scope_uri = scope.artifact_uri
        elif encoded_scope != run_state.scope_bytes:
            raise ExperimentError("enumerator changed scope between pages")
        if not page.candidates and not page.complete:
            raise ExperimentError(
                "incomplete enumerator page made no candidate progress"
            )
        if len(page.candidates) > page_size:
            raise ExperimentError("enumerator returned more candidates than requested")
        return page

    def _collect_unique_candidates(
        self,
        run_state: _EnumerationRunState,
        request: SearchEnumerateRequest,
        page: PluginEnumerationPage,
        remaining_wall: float,
    ) -> tuple[list[str], list[str]]:
        """Validate, canonicalize, and deduplicate candidates from one page."""

        selected_uris: list[str] = []
        selected_keys: list[str] = []
        manifest = self.plugins.get(request.plugin_id)
        self.store.get_descriptor(
            manifest.semantics_uri,
            expected_kind="semantics",
        )
        for payload in page.candidates:
            normalized_payload = self.schemas.validate(
                manifest.candidate_schema_uri,
                payload,
            )
            candidate = self.store.put(
                schema_uri=manifest.candidate_schema_uri,
                semantics_uri=manifest.semantics_uri,
                payload=normalized_payload,
                summary="enumerated candidate",
            )
            selected_uri = candidate.artifact_uri
            canonical_key = candidate.object_digest
            if request.quotient_by_isomorphism:
                canonical = self.structures.canonicalize(
                    structure_uri=candidate.artifact_uri,
                    plugin_id=request.plugin_id,
                    wall_seconds=max(1, int(remaining_wall)),
                )
                if (
                    canonical.result.execution.status != ExecutionStatus.COMPLETED
                    or canonical.result.input.status != InputStatus.ACCEPTED
                    or canonical.canonical_uri is None
                    or canonical.canonical_key is None
                ):
                    raise ExperimentError(
                        canonical.result.execution.detail
                        or "; ".join(canonical.result.input.errors)
                        or "canonicalization failed"
                    )
                selected_uri = canonical.canonical_uri
                canonical_key = canonical.canonical_key
            run_state.raw_candidates += 1
            if canonical_key in run_state.seen_keys:
                run_state.duplicate_candidates += 1
                continue
            run_state.seen_keys.add(canonical_key)
            run_state.unique_candidates += 1
            selected_uris.append(selected_uri)
            selected_keys.append(canonical_key)
        return selected_uris, selected_keys

    def _archive_evaluation_page(
        self,
        run_state: _EnumerationRunState,
        request: SearchEnumerateRequest,
        selected_uris: list[str],
        selected_keys: list[str],
        experiment_uri: str,
        started: float,
    ) -> None:
        """Evaluate and durably archive one page of unique candidates."""

        if not selected_uris:
            return
        evaluation = self.evaluation.evaluate_batch(
            claim_uri=request.claim_uri,
            candidate_uris=tuple(selected_uris),
            plugin_id=request.plugin_id,
            profile=request.profile,
            seed=request.seed,
            wall_seconds=max(
                1,
                int(request.budget.wall_seconds - (time.monotonic() - started)),
            ),
        )
        require_complete_evaluation_batch(evaluation, selected_uris)
        run_state.evaluated_candidates += len(selected_uris)
        evaluation_artifact = self._put_internal_artifact(
            schema_uri=self.evaluation_schema_uri,
            payload=evaluation.model_dump(mode="json"),
            parents=(request.claim_uri,),
            summary="untrusted enumeration evaluation batch",
        )
        archive_page = EnumerationArchive(
            experiment_uri=experiment_uri,
            page_index=len(run_state.page_uris),
            candidate_uris=tuple(selected_uris),
            evaluation_uris=tuple(
                evaluation_artifact.artifact_uri for _ in selected_uris
            ),
            canonical_keys=tuple(selected_keys),
        )
        stored_page = self._put_internal_artifact(
            schema_uri=self.archive_page_schema_uri,
            payload=archive_page.model_dump(mode="json"),
            parents=(
                evaluation_artifact.artifact_uri,
                *selected_uris,
                *run_state.page_uris[-1:],
            ),
            summary="enumeration archive page",
        )
        run_state.page_uris.append(stored_page.artifact_uri)

    def _commit_progress_and_check(
        self,
        run_state: _EnumerationRunState,
        experiment_uri: str,
        started: float,
        request: SearchEnumerateRequest,
        page: PluginEnumerationPage,
    ) -> bool:
        """Commit durable progress and return True when the run should stop."""

        accounting = run_state.accounting()
        current = self.inspect(experiment_uri)
        progress_committed = self._replace_running(
            _updated(
                current,
                updated_at=_now(),
                scope_uri=run_state.scope_uri,
                archive_page_uris=tuple(run_state.page_uris),
                accounting=accounting,
            )
        )
        if not progress_committed:
            self._finish(
                experiment_uri,
                state=ExperimentState.CANCELLED,
                stop_reason=EnumerationStopReason.CANCELLED,
                started=started,
                request=request,
                scope_uri=run_state.scope_uri,
                page_uris=run_state.page_uris,
                complete=False,
                accounting=accounting,
                detail="experiment cancelled",
            )
            return True
        if time.monotonic() - started >= request.budget.wall_seconds:
            self._finish(
                experiment_uri,
                state=ExperimentState.TIMEOUT,
                stop_reason=EnumerationStopReason.WALL_TIME_LIMIT,
                started=started,
                request=request,
                scope_uri=run_state.scope_uri,
                page_uris=run_state.page_uris,
                complete=False,
                accounting=accounting,
                detail="experiment wall-clock budget exhausted",
            )
            return True
        if page.complete:
            self._finish(
                experiment_uri,
                state=ExperimentState.COMPLETED,
                stop_reason=EnumerationStopReason.COMPLETE,
                started=started,
                request=request,
                scope_uri=run_state.scope_uri,
                page_uris=run_state.page_uris,
                complete=True,
                accounting=accounting,
                detail="enumerator reported complete bounded scope",
            )
            return True
        if page.next_cursor is None:
            raise ExperimentError("incomplete enumerator page requires next_cursor")
        if run_state.cursor is not None and canonicalize_json(
            page.next_cursor
        ) == canonicalize_json(run_state.cursor):
            raise ExperimentError("enumerator cursor did not advance")
        run_state.cursor = page.next_cursor
        return False

    def _run_enumeration(self, experiment_uri: str) -> None:
        snapshot = self.inspect(experiment_uri)
        request = snapshot.request
        started = time.monotonic()
        run_state = _EnumerationRunState()

        try:
            enumerator = self.plugins.resolve(
                request.plugin_id,
                CapabilityName.CANDIDATE_ENUMERATOR,
            )
            evaluator = self.plugins.resolve(
                request.plugin_id,
                CapabilityName.EVALUATOR,
            )
            canonicalizer = (
                self.plugins.resolve(
                    request.plugin_id,
                    CapabilityName.CANONICALIZER,
                )
                if request.quotient_by_isomorphism
                else None
            )
            if not self._mark_running(
                experiment_uri,
                enumerator_digest=enumerator.implementation_digest,
                canonicalizer_digest=(
                    canonicalizer.implementation_digest
                    if canonicalizer is not None
                    else None
                ),
                evaluator_digest=evaluator.implementation_digest,
            ):
                self._finish(
                    experiment_uri,
                    state=ExperimentState.CANCELLED,
                    stop_reason=EnumerationStopReason.CANCELLED,
                    started=started,
                    request=request,
                    scope_uri=None,
                    page_uris=[],
                    complete=False,
                    accounting=EnumerationAccounting(),
                    detail="experiment cancelled before execution",
                )
                return

            while True:
                remaining_wall = request.budget.wall_seconds - (
                    time.monotonic() - started
                )
                if self._check_enumeration_termination(
                    run_state, experiment_uri, started, request, remaining_wall
                ):
                    return
                page = self._run_enumerator_page(
                    run_state,
                    experiment_uri,
                    started,
                    request,
                    enumerator,
                    remaining_wall,
                )
                if page is None:
                    return
                selected_uris, selected_keys = self._collect_unique_candidates(
                    run_state, request, page, remaining_wall
                )
                self._archive_evaluation_page(
                    run_state,
                    request,
                    selected_uris,
                    selected_keys,
                    experiment_uri,
                    started,
                )
                if self._commit_progress_and_check(
                    run_state, experiment_uri, started, request, page
                ):
                    return
        except (
            ExperimentError,
            PluginRegistryError,
            SchemaRegistryError,
            StorageError,
            ValidationError,
            ValueError,
        ) as exc:
            detail = _enumeration_failure_detail(exc, experiment_uri)
            self._finish_if_possible(
                experiment_uri,
                started=started,
                scope_uri=run_state.scope_uri,
                page_uris=run_state.page_uris,
                accounting=run_state.accounting(),
                detail=detail,
            )
        finally:
            with self._thread_lock:
                self._threads.pop(experiment_uri, None)

    def _finish(
        self,
        experiment_uri: str,
        *,
        state: ExperimentState,
        stop_reason: EnumerationStopReason,
        started: float,
        request: SearchEnumerateRequest,
        scope_uri: str | None,
        page_uris: list[str],
        complete: bool,
        accounting: EnumerationAccounting,
        detail: str,
    ) -> None:
        manifest = EnumerationArchiveManifest(
            experiment_uri=experiment_uri,
            page_uris=tuple(page_uris),
            accounting=accounting,
        )
        archive = self._put_internal_artifact(
            schema_uri=self.archive_manifest_schema_uri,
            payload=manifest.model_dump(mode="json"),
            parents=(
                *((scope_uri,) if scope_uri is not None else ()),
                *page_uris[-1:],
            ),
            summary="enumeration archive manifest",
        )
        current = self.inspect(experiment_uri)
        terminal = _updated(
            current,
            state=state,
            updated_at=_now(),
            stop_reason=stop_reason,
            enumerator_reported_complete=complete,
            coverage=(
                Coverage.EXHAUSTIVE
                if complete
                and state == ExperimentState.COMPLETED
                and stop_reason == EnumerationStopReason.COMPLETE
                else Coverage.BOUNDED
            ),
            verification=Verification.UNVERIFIED,
            scope_uri=scope_uri,
            archive_uri=archive.artifact_uri,
            archive_page_uris=tuple(page_uris),
            accounting=accounting,
            detail=(f"{detail}; runtime_ms={int((time.monotonic() - started) * 1000)}"),
        )
        self._commit_terminal(terminal)

    def _finish_if_possible(
        self,
        experiment_uri: str,
        *,
        started: float,
        scope_uri: str | None,
        page_uris: list[str],
        accounting: EnumerationAccounting,
        detail: str,
    ) -> None:
        try:
            current = self.inspect(experiment_uri)
            if current.state in _TERMINAL_STATES:
                return
            self._finish(
                experiment_uri,
                state=ExperimentState.ERROR,
                stop_reason=EnumerationStopReason.ERROR,
                started=started,
                request=current.request,
                scope_uri=scope_uri,
                page_uris=page_uris,
                complete=False,
                accounting=accounting,
                detail=detail,
            )
        except (
            ExperimentError,
            StorageError,
            SchemaRegistryError,
            ValidationError,
        ) as exc:
            self._record_terminal_persistence_failure(
                experiment_uri,
                started=started,
                scope_uri=scope_uri,
                page_uris=page_uris,
                accounting=accounting,
                detail=detail,
                error=exc,
            )

    def _record_terminal_persistence_failure(
        self,
        experiment_uri: str,
        *,
        started: float,
        scope_uri: str | None,
        page_uris: list[str],
        accounting: EnumerationAccounting,
        detail: str,
        error: Exception,
    ) -> None:
        """Fail closed when the terminal archive cannot be committed."""

        _LOGGER.warning(
            "enumeration terminal archive persistence failed",
            exc_info=error,
        )
        with self.store.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT snapshot_json
                FROM experiments
                WHERE experiment_uri = ?
                """,
                (experiment_uri,),
            ).fetchone()
            if row is None:
                _LOGGER.warning(
                    "experiment not found: %s",
                    experiment_uri,
                )
                raise ExperimentNotFoundError(_EXPERIMENT_NOT_FOUND)
            current = _decode_experiment_snapshot(row["snapshot_json"], experiment_uri)
            if current.state in _TERMINAL_STATES:
                return
            terminal = _updated(
                current,
                state=ExperimentState.ERROR,
                updated_at=_now(),
                stop_reason=EnumerationStopReason.ERROR,
                enumerator_reported_complete=False,
                coverage=Coverage.BOUNDED,
                verification=Verification.UNVERIFIED,
                scope_uri=scope_uri,
                archive_uri=None,
                archive_page_uris=tuple(page_uris),
                accounting=accounting,
                detail=(
                    "Jacobian could not save the final experiment archive. "
                    "The experiment remains unverified. Check state-directory space "
                    "and permissions, then inspect the experiment."
                ),
            )
            connection.execute(
                """
                UPDATE experiments
                SET state = ?, snapshot_json = ?
                WHERE experiment_uri = ?
                """,
                (
                    terminal.state.value,
                    canonicalize_json(terminal.model_dump(mode="json")),
                    experiment_uri,
                ),
            )

    def _cancel_requested(self, experiment_uri: str) -> bool:
        return self.inspect(experiment_uri).state == ExperimentState.CANCEL_REQUESTED

    def _mark_running(
        self,
        experiment_uri: str,
        *,
        enumerator_digest: str,
        canonicalizer_digest: str | None,
        evaluator_digest: str,
    ) -> bool:
        with self.store.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT snapshot_json
                FROM experiments
                WHERE experiment_uri = ?
                """,
                (experiment_uri,),
            ).fetchone()
            if row is None:
                _LOGGER.warning("experiment not found: %s", experiment_uri)
                raise ExperimentNotFoundError(_EXPERIMENT_NOT_FOUND)
            snapshot = _decode_experiment_snapshot(row["snapshot_json"], experiment_uri)
            if snapshot.state == ExperimentState.CANCEL_REQUESTED:
                return False
            if snapshot.state != ExperimentState.PENDING:
                raise ExperimentError(
                    f"cannot start experiment from {snapshot.state.value}"
                )
            running = _updated(
                snapshot,
                state=ExperimentState.RUNNING,
                updated_at=_now(),
                enumerator_digest=enumerator_digest,
                canonicalizer_digest=canonicalizer_digest,
                evaluator_digest=evaluator_digest,
            )
            connection.execute(
                """
                UPDATE experiments
                SET state = ?, snapshot_json = ?
                WHERE experiment_uri = ?
                """,
                (
                    running.state.value,
                    canonicalize_json(running.model_dump(mode="json")),
                    experiment_uri,
                ),
            )
        return True

    def _write_new(self, snapshot: ExperimentSnapshot) -> None:
        with self.store.connection() as connection:
            connection.execute(
                """
                INSERT INTO experiments (
                    experiment_uri, state, snapshot_json
                ) VALUES (?, ?, ?)
                """,
                (
                    snapshot.experiment_uri,
                    snapshot.state.value,
                    canonicalize_json(snapshot.model_dump(mode="json")),
                ),
            )

    def _replace_running(self, snapshot: ExperimentSnapshot) -> bool:
        """Commit progress unless cancellation won the transaction race."""

        with self.store.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT snapshot_json
                FROM experiments
                WHERE experiment_uri = ?
                """,
                (snapshot.experiment_uri,),
            ).fetchone()
            if row is None:
                _LOGGER.warning(
                    "experiment not found: %s",
                    snapshot.experiment_uri,
                )
                raise ExperimentNotFoundError(_EXPERIMENT_NOT_FOUND)
            current = _decode_experiment_snapshot(
                row["snapshot_json"], snapshot.experiment_uri
            )
            if current.state == ExperimentState.CANCEL_REQUESTED:
                return False
            if current.state != ExperimentState.RUNNING:
                raise ExperimentError(
                    f"cannot write progress from {current.state.value}"
                )
            connection.execute(
                """
                UPDATE experiments
                SET state = ?, snapshot_json = ?
                WHERE experiment_uri = ?
                """,
                (
                    snapshot.state.value,
                    canonicalize_json(snapshot.model_dump(mode="json")),
                    snapshot.experiment_uri,
                ),
            )
        return True

    def _commit_terminal(self, snapshot: ExperimentSnapshot) -> None:
        """Atomically commit a terminal snapshot while honoring cancellation."""

        with self.store.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT snapshot_json
                FROM experiments
                WHERE experiment_uri = ?
                """,
                (snapshot.experiment_uri,),
            ).fetchone()
            if row is None:
                _LOGGER.warning("experiment not found: %s", snapshot.experiment_uri)
                raise ExperimentError(_EXPERIMENT_NOT_FOUND)
            current = _decode_experiment_snapshot(
                row["snapshot_json"], snapshot.experiment_uri
            )
            terminal = snapshot
            if current.state == ExperimentState.CANCEL_REQUESTED:
                terminal = _updated(
                    snapshot,
                    state=ExperimentState.CANCELLED,
                    stop_reason=EnumerationStopReason.CANCELLED,
                    enumerator_reported_complete=False,
                    coverage=Coverage.BOUNDED,
                    verification=Verification.UNVERIFIED,
                    detail=f"experiment cancelled; {snapshot.detail}",
                )
            elif current.state in _TERMINAL_STATES:
                raise ExperimentError(
                    f"experiment is already terminal: {current.state.value}"
                )
            connection.execute(
                """
                UPDATE experiments
                SET state = ?, snapshot_json = ?
                WHERE experiment_uri = ?
                """,
                (
                    terminal.state.value,
                    canonicalize_json(terminal.model_dump(mode="json")),
                    terminal.experiment_uri,
                ),
            )
