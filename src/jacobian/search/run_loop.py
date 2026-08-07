"""Search experiment execution and iteration orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from jacobian.contracts.discovery import ExperimentState
from jacobian.contracts.plugins import PluginManifest
from jacobian.contracts.results import ExecutionStatus
from jacobian.contracts.search import (
    PluginProposalResponse,
    PluginRefinementResponse,
    SearchAccounting,
    SearchArchivePage,
    SearchBudget,
    SearchCandidateRecord,
    SearchCheckpoint,
    SearchExperimentSnapshot,
    SearchRunRequest,
    SearchStopReason,
)
from jacobian.contracts.witness_search import WitnessSearchStatus
from jacobian.evaluation import require_complete_evaluation_batch
from jacobian.plugins.registry import (
    PluginRegistryError,
    ResolvedCapability,
)
from jacobian.schema_registry import SchemaRegistryError
from jacobian.search._helpers import (
    _deduplicate_nominations,
    _digest,
    _is_verified_counterexample,
    _now,
    _record_parents,
    _require_remaining_seconds,
    _search_failure_detail,
    _updated_accounting,
    _updated_snapshot,
    _used_wall_ms,
)
from jacobian.search.errors import SearchError, _SearchBudgetExhaustedError
from jacobian.search.recovery import RestoredSearchProgress, restore_search_progress
from jacobian.storage.errors import StorageError
from jacobian.storage.models import StoredArtifact

if TYPE_CHECKING:
    from jacobian.search.service import SearchService


@dataclass(slots=True)
class _SearchIterationState:
    """All durable and strategy state needed by one search coordinator."""

    service: SearchService
    experiment_uri: str
    snapshot: SearchExperimentSnapshot
    request: SearchRunRequest
    claim: StoredArtifact
    manifest: PluginManifest
    semantics: StoredArtifact
    proposer: ResolvedCapability
    refiner: ResolvedCapability
    evaluator_digest: str
    strategy_state: dict[str, Any]
    page_uris: list[str]
    seen_uris: set[str]
    nominated_uris: set[str]
    accounting: SearchAccounting
    partial_accounting: SearchAccounting
    started: float


@dataclass(frozen=True, slots=True)
class _CandidateEvaluation:
    """Counters and records produced by one evaluated candidate batch."""

    proposed: int
    unique: int
    duplicates: int
    evaluated: int
    attacked: int
    verified_counterexamples: int
    records: tuple[SearchCandidateRecord, ...]


def execute_search(service: SearchService, experiment_uri: str) -> None:
    """Execute one durable search experiment until a terminal state."""

    started = service._clock()
    accounting = SearchAccounting()
    partial_accounting = accounting
    state: _SearchIterationState | None = None
    try:
        state = _initialize_search(service, experiment_uri, started, accounting)
        if state is None:
            return
        while _run_iteration(state):
            pass
        accounting = state.accounting
        partial_accounting = state.partial_accounting
        started = state.started
    except _SearchBudgetExhaustedError as exc:
        if state is not None:
            accounting = state.accounting
            partial_accounting = state.partial_accounting
            started = state.started
        service._finish_if_possible(
            experiment_uri,
            state=ExperimentState.TIMEOUT,
            stop_reason=SearchStopReason.WALL_TIME_LIMIT,
            detail=str(exc),
            wall_time_ms=_used_wall_ms(accounting, started, service._clock),
            accounting_override=partial_accounting,
        )
    except (
        SearchError,
        PluginRegistryError,
        SchemaRegistryError,
        StorageError,
        ValidationError,
        ValueError,
    ) as exc:
        if state is not None:
            accounting = state.accounting
            partial_accounting = state.partial_accounting
            started = state.started
        service._finish_if_possible(
            experiment_uri,
            state=ExperimentState.ERROR,
            stop_reason=SearchStopReason.ERROR,
            detail=_search_failure_detail(exc, experiment_uri),
            wall_time_ms=_used_wall_ms(accounting, started, service._clock),
            accounting_override=partial_accounting,
        )
    finally:
        with service._thread_lock:
            service._threads.pop(experiment_uri, None)


def _initialize_search(
    service: SearchService,
    experiment_uri: str,
    started: float,
    accounting: SearchAccounting,
) -> _SearchIterationState | None:
    snapshot = service.inspect(experiment_uri)
    transition = service._mark_running(snapshot)
    if transition == ExperimentState.PAUSED:
        return None
    if transition == ExperimentState.CANCEL_REQUESTED:
        service._finish(
            experiment_uri,
            state=ExperimentState.CANCELLED,
            stop_reason=SearchStopReason.CANCELLED,
            strategy_complete=False,
            detail="search cancelled before execution",
            wall_time_ms=_used_wall_ms(accounting, started, service._clock),
        )
        return None

    snapshot = service.inspect(experiment_uri)
    request = snapshot.request
    proposer, refiner, evaluator_digest = service._resolve_strategy(snapshot)
    restored = _restore_progress(service, snapshot)
    claim = service.store.get(request.claim_uri)
    manifest = service.plugins.get(request.plugin_id)
    semantics = service.store.get(manifest.semantics_uri)
    return _SearchIterationState(
        service=service,
        experiment_uri=experiment_uri,
        snapshot=snapshot,
        request=request,
        claim=claim,
        manifest=manifest,
        semantics=semantics,
        proposer=proposer,
        refiner=refiner,
        evaluator_digest=evaluator_digest,
        strategy_state=restored.strategy_state,
        page_uris=restored.page_uris,
        seen_uris=restored.seen_uris,
        nominated_uris=restored.nominated_uris,
        accounting=restored.accounting,
        partial_accounting=restored.accounting,
        started=started,
    )


def _restore_progress(
    service: SearchService,
    snapshot: SearchExperimentSnapshot,
) -> RestoredSearchProgress:
    return restore_search_progress(
        snapshot,
        store=service.store,
        semantics_uri=service.semantics_uri,
        archive_page_schema_uri=service.archive_page_schema_uri,
        checkpoint_schema_uri=service.checkpoint_schema_uri,
    )


def _run_iteration(state: _SearchIterationState) -> bool:
    budget = state.snapshot.effective_budget
    total_wall_ms = _used_wall_ms(
        state.accounting,
        state.started,
        state.service._clock,
    )
    if state.service._budget_exhausted(
        state.experiment_uri,
        state.accounting,
        budget,
        wall_time_ms=total_wall_ms,
    ):
        return False
    proposal = _run_proposer(state, budget, total_wall_ms)
    if proposal is None:
        return False
    evaluation = _evaluate_candidates(state, proposal, budget)
    max_additional_lineage_parents = _max_additional_nomination_parents(
        state, evaluation.records
    )
    refinement = _run_refiner(
        state,
        proposal,
        evaluation.records,
        budget,
        max_additional_lineage_parents=max_additional_lineage_parents,
    )
    if refinement is None:
        return False
    return _persist_iteration(
        state,
        proposal,
        refinement,
        evaluation,
        budget,
        max_additional_lineage_parents=max_additional_lineage_parents,
    )


def _run_proposer(
    state: _SearchIterationState,
    budget: SearchBudget,
    total_wall_ms: int,
) -> PluginProposalResponse | None:
    remaining_candidates = budget.candidates_max - state.accounting.proposed_candidates
    batch_size = min(budget.batch_size, remaining_candidates)
    remaining_seconds = max(0.001, budget.wall_seconds - total_wall_ms / 1000)
    request = {
        "request_version": "1",
        "claim": state.claim.payload,
        "state": state.strategy_state,
        "batch_size": batch_size,
        "seed": state.request.seed,
        "remaining_budget": {
            "candidates": remaining_candidates,
            "iterations": budget.iterations_max - state.accounting.iterations,
            "wall_ms": max(1, int(remaining_seconds * 1000)),
        },
        "bindings": {
            "claim_digest": state.claim.manifest.object_digest,
            "semantics_digest": state.semantics.manifest.object_digest,
            "plugin_id": state.request.plugin_id,
            "request_digest": state.snapshot.request_digest,
        },
    }
    execution = state.service.executor.run(
        entrypoint=state.proposer.descriptor.entrypoint,
        implementation_digest=state.proposer.implementation_digest,
        request=request,
        timeout_seconds=remaining_seconds,
    )
    state.service._record_operation(
        state.experiment_uri,
        event_type="PROPOSER_COMPLETED",
        payload={
            "iteration": state.accounting.iterations + 1,
            "status": execution.status.value,
            "implementation_digest": state.proposer.implementation_digest,
            "request_digest": _digest(request),
            "output_digest": (
                _digest(execution.output) if execution.output is not None else None
            ),
            "runtime_ms": execution.runtime_ms,
            "detail": execution.detail,
        },
    )
    if execution.status != ExecutionStatus.COMPLETED:
        state.service._finish_execution_failure(
            state.experiment_uri,
            execution.status,
            execution.detail or "proposer execution failed",
            wall_time_ms=_used_wall_ms(
                state.accounting,
                state.started,
                state.service._clock,
            ),
        )
        return None
    proposal = PluginProposalResponse.model_validate(execution.output)
    if len(proposal.candidates) > batch_size:
        raise SearchError("proposer returned more candidates than authorized")
    return proposal


def _evaluate_candidates(
    state: _SearchIterationState,
    proposal: PluginProposalResponse,
    budget: SearchBudget,
) -> _CandidateEvaluation:
    selected, proposed, unique, duplicates = _select_candidates(state, proposal, budget)
    evaluated = state.accounting.evaluated_candidates
    attacked = state.accounting.attacked_candidates
    verified = state.accounting.verified_counterexamples
    if not selected:
        return _CandidateEvaluation(
            proposed=proposed,
            unique=unique,
            duplicates=duplicates,
            evaluated=evaluated,
            attacked=attacked,
            verified_counterexamples=verified,
            records=(),
        )

    remaining_seconds = _require_remaining_seconds(
        budget,
        state.accounting,
        state.started,
        state.service._clock,
    )
    evaluation = state.service.evaluation.evaluate_batch(
        claim_uri=state.request.claim_uri,
        candidate_uris=tuple(selected),
        plugin_id=state.request.plugin_id,
        profile=state.request.profile,
        seed=state.request.seed,
        wall_seconds=remaining_seconds,
    )
    require_complete_evaluation_batch(evaluation, selected)
    evaluated += len(selected)
    state.partial_accounting = _updated_accounting(
        state.partial_accounting,
        evaluated_candidates=evaluated,
    )
    evaluation_artifact = state.service._put_internal_artifact(
        schema_uri=state.service.evaluation_schema_uri,
        payload=evaluation.model_dump(mode="json"),
        parents=(state.request.claim_uri, *selected),
        summary="untrusted search evaluation batch",
    )
    state.service._record_operation(
        state.experiment_uri,
        event_type="EVALUATION_COMMITTED",
        payload={
            "iteration": state.accounting.iterations + 1,
            "candidate_uris": selected,
            "evaluation_uri": evaluation_artifact.artifact_uri,
            "evaluator_digest": state.evaluator_digest,
            "status": evaluation.execution.status.value,
            "runtime_ms": evaluation.execution.runtime_ms,
        },
    )
    records: list[SearchCandidateRecord] = []
    for candidate_uri in selected:
        record, was_attacked, was_verified = _evaluate_candidate(
            state,
            candidate_uri=candidate_uri,
            evaluation_uri=evaluation_artifact.artifact_uri,
            budget=budget,
        )
        records.append(record)
        attacked += int(was_attacked)
        verified += int(was_verified)
    return _CandidateEvaluation(
        proposed=proposed,
        unique=unique,
        duplicates=duplicates,
        evaluated=evaluated,
        attacked=attacked,
        verified_counterexamples=verified,
        records=tuple(records),
    )


def _select_candidates(
    state: _SearchIterationState,
    proposal: PluginProposalResponse,
    budget: SearchBudget,
) -> tuple[list[str], int, int, int]:
    selected: list[str] = []
    proposed = state.accounting.proposed_candidates + len(proposal.candidates)
    duplicates = state.accounting.duplicate_candidates
    unique = state.accounting.unique_candidates
    for payload in proposal.candidates:
        normalized = state.service.schemas.validate(
            state.manifest.candidate_schema_uri,
            payload,
        )
        candidate = state.service.store.put(
            schema_uri=state.manifest.candidate_schema_uri,
            semantics_uri=state.manifest.semantics_uri,
            payload=normalized,
            parents=(state.request.claim_uri, state.request.plugin_id),
            summary="search candidate proposed by untrusted strategy",
        )
        if candidate.artifact_uri in state.seen_uris:
            duplicates += 1
            state.partial_accounting = _updated_accounting(
                state.accounting,
                proposed_candidates=unique + duplicates,
                unique_candidates=unique,
                duplicate_candidates=duplicates,
            )
            continue
        state.seen_uris.add(candidate.artifact_uri)
        selected.append(candidate.artifact_uri)
        unique += 1
        state.partial_accounting = _updated_accounting(
            state.accounting,
            proposed_candidates=unique + duplicates,
            unique_candidates=unique,
            duplicate_candidates=duplicates,
        )
    if len(selected) > budget.batch_size:
        raise SearchError("selected candidate count exceeds the effective batch size")
    return selected, proposed, unique, duplicates


def _evaluate_candidate(
    state: _SearchIterationState,
    *,
    candidate_uri: str,
    evaluation_uri: str,
    budget: SearchBudget,
) -> tuple[SearchCandidateRecord, bool, bool]:
    witness_uri: str | None = None
    verification_record_uri: str | None = None
    counterexample_verified = False
    detail = ""
    was_attacked = state.request.witness_role is not None
    if state.request.witness_role is not None:
        remaining_seconds = _require_remaining_seconds(
            budget,
            state.accounting,
            state.started,
            state.service._clock,
        )
        state.partial_accounting = _updated_accounting(
            state.partial_accounting,
            attacked_candidates=state.partial_accounting.attacked_candidates + 1,
        )
        witness_result = state.service.witnesses.find(
            claim_uri=state.request.claim_uri,
            candidate_uri=candidate_uri,
            plugin_id=state.request.plugin_id,
            witness_role=state.request.witness_role,
            wall_seconds=remaining_seconds,
        )
        witness_uri = witness_result.witness_uri
        detail = witness_result.detail
        if (
            witness_result.status == WitnessSearchStatus.FOUND
            and witness_uri is not None
        ):
            checker_id = state.request.counterexample_checker_id
            if checker_id is None:
                raise SearchError(
                    "A counterexample witness was found, but no checker was supplied. "
                    "Set counterexample_checker_id from the reference contract and retry."
                )
            checker_remaining = _require_remaining_seconds(
                budget,
                state.accounting,
                state.started,
                state.service._clock,
            )
            verified = state.service.verification.verify_witness(
                claim_uri=state.request.claim_uri,
                candidate_uri=candidate_uri,
                witness_uri=witness_uri,
                checker_id=checker_id,
                timeout_seconds=checker_remaining,
            )
            if _is_verified_counterexample(verified):
                counterexample_verified = True
                verification_record_uri = verified.verification_record_uri
                state.partial_accounting = _updated_accounting(
                    state.partial_accounting,
                    verified_counterexamples=(
                        state.partial_accounting.verified_counterexamples + 1
                    ),
                )
        state.service._record_operation(
            state.experiment_uri,
            event_type="COUNTEREXAMPLE_ATTEMPTED",
            payload={
                "iteration": state.accounting.iterations + 1,
                "candidate_uri": candidate_uri,
                "status": witness_result.status.value,
                "witness_uri": witness_uri,
                "verification_record_uri": verification_record_uri,
                "verified": counterexample_verified,
            },
        )
    return (
        SearchCandidateRecord(
            candidate_uri=candidate_uri,
            evaluation_uri=evaluation_uri,
            witness_uri=witness_uri,
            verification_record_uri=verification_record_uri,
            counterexample_verified=counterexample_verified,
            detail=detail,
        ),
        was_attacked,
        counterexample_verified,
    )


def _run_refiner(
    state: _SearchIterationState,
    proposal: PluginProposalResponse,
    records: tuple[SearchCandidateRecord, ...],
    budget: SearchBudget,
    *,
    max_additional_lineage_parents: int,
) -> PluginRefinementResponse | None:
    request = {
        "request_version": "1",
        "claim": state.claim.payload,
        "state": proposal.state,
        "feedback": [record.model_dump(mode="json") for record in records],
        "strategy_reported_complete": proposal.complete,
        "seed": state.request.seed,
        "max_additional_lineage_parents": max_additional_lineage_parents,
        "bindings": {
            "claim_digest": state.claim.manifest.object_digest,
            "semantics_digest": state.semantics.manifest.object_digest,
            "plugin_id": state.request.plugin_id,
            "request_digest": state.snapshot.request_digest,
        },
    }
    remaining_seconds = _require_remaining_seconds(
        budget,
        state.accounting,
        state.started,
        state.service._clock,
    )
    execution = state.service.executor.run(
        entrypoint=state.refiner.descriptor.entrypoint,
        implementation_digest=state.refiner.implementation_digest,
        request=request,
        timeout_seconds=remaining_seconds,
    )
    state.service._record_operation(
        state.experiment_uri,
        event_type="REFINER_COMPLETED",
        payload={
            "iteration": state.accounting.iterations + 1,
            "status": execution.status.value,
            "implementation_digest": state.refiner.implementation_digest,
            "request_digest": _digest(request),
            "output_digest": (
                _digest(execution.output) if execution.output is not None else None
            ),
            "runtime_ms": execution.runtime_ms,
            "detail": execution.detail,
            "feedback_records": [record.model_dump(mode="json") for record in records],
        },
    )
    if execution.status != ExecutionStatus.COMPLETED:
        state.service._finish_execution_failure(
            state.experiment_uri,
            execution.status,
            execution.detail or "refiner execution failed",
            wall_time_ms=_used_wall_ms(
                state.accounting,
                state.started,
                state.service._clock,
            ),
            accounting_override=state.partial_accounting,
        )
        return None
    refinement = PluginRefinementResponse.model_validate(execution.output)
    for nomination in refinement.nominations:
        if nomination.candidate_uri not in state.seen_uris:
            raise SearchError("refiner nominated a candidate outside this search")
    return refinement


def _max_additional_nomination_parents(
    state: _SearchIterationState,
    records: tuple[SearchCandidateRecord, ...],
) -> int:
    required_parents = {
        state.request.claim_uri,
        state.request.plugin_id,
        *_record_parents(list(records), ()),
    }
    return max(0, state.service.store.limits.max_parents - len(required_parents))


def _persist_iteration(
    state: _SearchIterationState,
    proposal: PluginProposalResponse,
    refinement: PluginRefinementResponse,
    evaluation: _CandidateEvaluation,
    budget: SearchBudget,
    *,
    max_additional_lineage_parents: int,
) -> bool:
    nominations = tuple(
        nomination
        for nomination in _deduplicate_nominations(refinement.nominations)
        if nomination.candidate_uri not in state.nominated_uris
    )
    record_parents = set(_record_parents(list(evaluation.records), ()))
    additional_nomination_parents = {
        nomination.candidate_uri for nomination in nominations
    } - record_parents
    if len(additional_nomination_parents) > max_additional_lineage_parents:
        raise SearchError(
            "refiner returned more nomination lineage than the archive can preserve"
        )
    state.nominated_uris.update(nomination.candidate_uri for nomination in nominations)
    next_accounting = SearchAccounting(
        proposed_candidates=evaluation.proposed,
        unique_candidates=evaluation.unique,
        duplicate_candidates=evaluation.duplicates,
        evaluated_candidates=evaluation.evaluated,
        attacked_candidates=evaluation.attacked,
        verified_counterexamples=evaluation.verified_counterexamples,
        iterations=state.accounting.iterations + 1,
        checkpoints=state.accounting.checkpoints + 1,
        nominations=state.accounting.nominations + len(nominations),
        wall_time_ms=_used_wall_ms(
            state.accounting,
            state.started,
            state.service._clock,
        ),
    )
    state.partial_accounting = next_accounting
    page = SearchArchivePage(
        experiment_uri=state.experiment_uri,
        request_digest=state.snapshot.request_digest,
        claim_uri=state.request.claim_uri,
        plugin_id=state.request.plugin_id,
        registry_snapshot_uri=state.snapshot.registry_snapshot_uri,
        iteration=next_accounting.iterations,
        proposer_digest=state.proposer.implementation_digest,
        refiner_digest=state.refiner.implementation_digest,
        evaluator_digest=state.evaluator_digest,
        records=evaluation.records,
        nominations=nominations,
    )
    page_parents = _record_parents(list(evaluation.records), nominations)
    stored_page = state.service._put_internal_artifact(
        schema_uri=state.service.archive_page_schema_uri,
        payload=page.model_dump(mode="json"),
        parents=(state.request.claim_uri, state.request.plugin_id, *page_parents),
        summary="search archive page",
    )
    state.page_uris.append(stored_page.artifact_uri)
    checkpoint = SearchCheckpoint(
        experiment_uri=state.experiment_uri,
        request_digest=state.snapshot.request_digest,
        iteration=next_accounting.iterations,
        state=refinement.state,
        latest_records=evaluation.records,
        nominations=nominations,
        accounting=next_accounting,
        effective_budget=budget,
        registry_snapshot_uri=state.snapshot.registry_snapshot_uri,
        proposer_digest=state.proposer.implementation_digest,
        refiner_digest=state.refiner.implementation_digest,
        evaluator_digest=state.evaluator_digest,
        environment_digest=state.snapshot.environment_digest,
        previous_checkpoint_uri=state.snapshot.checkpoint_uri,
    )
    checkpoint_parents = [stored_page.artifact_uri]
    if state.snapshot.checkpoint_uri is not None:
        checkpoint_parents.append(state.snapshot.checkpoint_uri)
    stored_checkpoint = state.service._put_internal_artifact(
        schema_uri=state.service.checkpoint_schema_uri,
        payload=checkpoint.model_dump(mode="json"),
        parents=tuple(checkpoint_parents),
        summary="immutable search checkpoint",
    )
    persisted_accounting = next_accounting.model_copy(
        update={
            "wall_time_ms": _used_wall_ms(
                state.accounting,
                state.started,
                state.service._clock,
            )
        }
    )
    current = state.service.inspect(state.experiment_uri)
    progress = _updated_snapshot(
        current,
        state=ExperimentState.RUNNING,
        updated_at=_now(),
        checkpoint_uri=stored_checkpoint.artifact_uri,
        archive_page_uris=tuple(state.page_uris),
        accounting=persisted_accounting,
        detail=proposal.detail or refinement.detail,
    )
    control_state = state.service._commit_progress(progress)
    state.snapshot = state.service.inspect(state.experiment_uri)
    state.strategy_state = refinement.state
    state.accounting = persisted_accounting
    state.partial_accounting = persisted_accounting
    state.started = state.service._clock()
    if control_state == ExperimentState.PAUSED:
        return False
    if control_state == ExperimentState.CANCEL_REQUESTED:
        state.service._finish(
            state.experiment_uri,
            state=ExperimentState.CANCELLED,
            stop_reason=SearchStopReason.CANCELLED,
            strategy_complete=False,
            detail="search cancelled",
            wall_time_ms=state.accounting.wall_time_ms,
        )
        return False
    if state.accounting.wall_time_ms >= budget.wall_seconds * 1000:
        state.service._finish(
            state.experiment_uri,
            state=ExperimentState.TIMEOUT,
            stop_reason=SearchStopReason.WALL_TIME_LIMIT,
            strategy_complete=False,
            detail="search wall-clock budget exhausted",
            wall_time_ms=state.accounting.wall_time_ms,
        )
        return False
    if proposal.complete:
        state.service._finish(
            state.experiment_uri,
            state=ExperimentState.COMPLETED,
            stop_reason=SearchStopReason.STRATEGY_COMPLETE,
            strategy_complete=True,
            detail="strategy reported completion",
            wall_time_ms=state.accounting.wall_time_ms,
        )
        return False
    return True
