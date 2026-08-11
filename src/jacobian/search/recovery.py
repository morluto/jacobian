"""Replay and validate immutable search progress."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jacobian.contracts.search import (
    SearchAccounting,
    SearchArchivePage,
    SearchCheckpoint,
    SearchExperimentSnapshot,
)
from jacobian.search._helpers import _record_parents
from jacobian.search.errors import SearchCorruptionError, SearchError
from jacobian.storage.repository import ArtifactRepository


@dataclass(frozen=True, slots=True)
class RestoredSearchProgress:
    """Validated opaque strategy state and its committed archive indexes."""

    strategy_state: dict[str, Any]
    page_uris: list[str]
    seen_uris: set[str]
    nominated_uris: set[str]
    accounting: SearchAccounting


def restore_search_progress(
    snapshot: SearchExperimentSnapshot,
    *,
    store: ArtifactRepository,
    semantics_uri: str,
    archive_page_schema_uri: str,
    checkpoint_schema_uri: str,
) -> RestoredSearchProgress:
    """Rebind immutable archive pages and checkpoint state to the snapshot.

    The snapshot's later wall-time sample may exceed the checkpoint payload
    because it includes checkpoint artifact persistence. All other identities,
    lineage, and accounting relationships must match exactly before opaque
    strategy state is returned to the execution loop.
    """

    page_uris = list(snapshot.archive_page_uris)
    seen_uris: set[str] = set()
    nominated_uris: set[str] = set()
    seen_page_uris: set[str] = set()
    verified_counterexamples = 0
    last_page: SearchArchivePage | None = None
    for iteration, page_uri in enumerate(page_uris, start=1):
        page, verified = _validate_archive_page(
            snapshot=snapshot,
            page_uri=page_uri,
            iteration=iteration,
            seen_uris=seen_uris,
            nominated_uris=nominated_uris,
            seen_page_uris=seen_page_uris,
            store=store,
            semantics_uri=semantics_uri,
            archive_page_schema_uri=archive_page_schema_uri,
        )
        verified_counterexamples += verified
        last_page = page
    if snapshot.checkpoint_uri is None:
        if page_uris or snapshot.accounting != SearchAccounting():
            raise SearchError("search progress is missing its checkpoint")
        return RestoredSearchProgress(
            strategy_state=snapshot.request.initial_state,
            page_uris=page_uris,
            seen_uris=seen_uris,
            nominated_uris=nominated_uris,
            accounting=snapshot.accounting,
        )
    if last_page is None:
        raise SearchError("checkpoint progress does not match the archive")
    state = _validate_checkpoint(
        snapshot=snapshot,
        page_uris=page_uris,
        seen_uris=seen_uris,
        nominated_uris=nominated_uris,
        verified_counterexamples=verified_counterexamples,
        last_page=last_page,
        store=store,
        semantics_uri=semantics_uri,
        checkpoint_schema_uri=checkpoint_schema_uri,
    )
    return RestoredSearchProgress(
        strategy_state=state,
        page_uris=page_uris,
        seen_uris=seen_uris,
        nominated_uris=nominated_uris,
        accounting=snapshot.accounting,
    )


def _validate_archive_page(
    *,
    snapshot: SearchExperimentSnapshot,
    page_uri: str,
    iteration: int,
    seen_uris: set[str],
    nominated_uris: set[str],
    seen_page_uris: set[str],
    store: ArtifactRepository,
    semantics_uri: str,
    archive_page_schema_uri: str,
) -> tuple[SearchArchivePage, int]:
    if page_uri in seen_page_uris:
        raise SearchError("search archive contains a repeated page")
    seen_page_uris.add(page_uri)
    page_artifact = store.get(page_uri)
    if (
        page_artifact.manifest.schema_uri != archive_page_schema_uri
        or page_artifact.manifest.semantics_uri != semantics_uri
    ):
        raise SearchError("archive page uses the wrong contract")
    page = SearchArchivePage.model_validate(page_artifact.payload)
    if (
        page.experiment_uri != snapshot.experiment_uri
        or page.request_digest != snapshot.request_digest
        or page.claim_uri != snapshot.request.claim_uri
        or page.plugin_id != snapshot.request.plugin_id
        or page.registry_snapshot_uri != snapshot.registry_snapshot_uri
        or page.proposer_digest != snapshot.proposer_digest
        or page.refiner_digest != snapshot.refiner_digest
        or page.evaluator_digest != snapshot.evaluator_digest
    ):
        raise SearchError("archive page identity does not match the search")
    if page.iteration != iteration:
        raise SearchError("archive page iteration sequence is invalid")
    required_parents = {
        snapshot.request.claim_uri,
        snapshot.request.plugin_id,
        *_record_parents(list(page.records), page.nominations),
    }
    if not required_parents.issubset(page_artifact.manifest.parents):
        raise SearchError("archive page is missing its declared lineage")
    verified = _validate_archive_contents(
        page,
        seen_uris=seen_uris,
        nominated_uris=nominated_uris,
    )
    return page, verified


def _validate_archive_contents(
    page: SearchArchivePage,
    *,
    seen_uris: set[str],
    nominated_uris: set[str],
) -> int:
    verified = 0
    for record in page.records:
        if record.candidate_uri in seen_uris:
            raise SearchError("candidate appears in multiple archive pages")
        seen_uris.add(record.candidate_uri)
        verified += int(record.counterexample_verified)
    for nomination in page.nominations:
        if nomination.candidate_uri not in seen_uris:
            raise SearchError("archive page nominates an unknown candidate")
        if nomination.candidate_uri in nominated_uris:
            raise SearchError("candidate is nominated more than once")
        nominated_uris.add(nomination.candidate_uri)
    return verified


def _validate_checkpoint(
    *,
    snapshot: SearchExperimentSnapshot,
    page_uris: list[str],
    seen_uris: set[str],
    nominated_uris: set[str],
    verified_counterexamples: int,
    last_page: SearchArchivePage,
    store: ArtifactRepository,
    semantics_uri: str,
    checkpoint_schema_uri: str,
) -> dict[str, Any]:
    if snapshot.checkpoint_uri is None:
        raise SearchCorruptionError("snapshot checkpoint URI is unexpectedly None")
    checkpoint_artifact = store.get(snapshot.checkpoint_uri)
    if (
        checkpoint_artifact.manifest.schema_uri != checkpoint_schema_uri
        or checkpoint_artifact.manifest.semantics_uri != semantics_uri
    ):
        raise SearchError("checkpoint uses the wrong contract")
    checkpoint = SearchCheckpoint.model_validate(checkpoint_artifact.payload)
    checkpoint_counts = checkpoint.accounting.model_copy(
        update={"wall_time_ms": snapshot.accounting.wall_time_ms}
    )
    if (
        checkpoint.experiment_uri != snapshot.experiment_uri
        or checkpoint.request_digest != snapshot.request_digest
        or checkpoint_counts != snapshot.accounting
        or checkpoint.accounting.wall_time_ms > snapshot.accounting.wall_time_ms
        or checkpoint.effective_budget != snapshot.effective_budget
        or checkpoint.registry_snapshot_uri != snapshot.registry_snapshot_uri
        or checkpoint.proposer_digest != snapshot.proposer_digest
        or checkpoint.refiner_digest != snapshot.refiner_digest
        or checkpoint.evaluator_digest != snapshot.evaluator_digest
        or checkpoint.environment_digest != snapshot.environment_digest
    ):
        raise SearchError("checkpoint identity does not match the search")
    if (
        checkpoint.iteration != len(page_uris)
        or checkpoint.accounting.iterations != len(page_uris)
        or checkpoint.accounting.checkpoints != len(page_uris)
        or checkpoint.latest_records != last_page.records
        or checkpoint.nominations != last_page.nominations
    ):
        raise SearchError("checkpoint progress does not match the archive")
    checkpoint_parents = {page_uris[-1]}
    if checkpoint.previous_checkpoint_uri is not None:
        checkpoint_parents.add(checkpoint.previous_checkpoint_uri)
    if not checkpoint_parents.issubset(checkpoint_artifact.manifest.parents):
        raise SearchError("checkpoint is missing its declared lineage")
    if (checkpoint.previous_checkpoint_uri is None) != (len(page_uris) == 1):
        raise SearchError("checkpoint predecessor does not match the archive")
    expected_attacks = (
        len(seen_uris) if snapshot.request.witness_role is not None else 0
    )
    if (
        snapshot.accounting.unique_candidates != len(seen_uris)
        or snapshot.accounting.evaluated_candidates != len(seen_uris)
        or snapshot.accounting.attacked_candidates != expected_attacks
        or snapshot.accounting.verified_counterexamples != verified_counterexamples
        or snapshot.accounting.nominations != len(nominated_uris)
    ):
        raise SearchError("search accounting does not match the archive")
    return checkpoint.state
