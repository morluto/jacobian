"""Durable, strategy-neutral search orchestration."""

from __future__ import annotations

import hashlib
import logging
import math
import platform
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from jacobian.canonical import canonicalize_json
from jacobian.contracts.discovery import ExperimentState
from jacobian.contracts.results import (
    Conclusion,
    ResultEnvelope,
    Verification,
)
from jacobian.contracts.search import (
    SearchAccounting,
    SearchBudget,
    SearchCandidateRecord,
    SearchExperimentSnapshot,
    SearchLifecycleEvent,
    SearchNomination,
)
from jacobian.plugins.registry import (
    PluginRegistryError,
)
from jacobian.search.errors import SearchError, _SearchBudgetExhaustedError
from jacobian.storage.errors import StorageError

_LOGGER = logging.getLogger(__name__)

_TERMINAL_STATES = {
    ExperimentState.COMPLETED,
    ExperimentState.CANCELLED,
    ExperimentState.TIMEOUT,
    ExperimentState.ERROR,
}
_SETTLED_STATES = _TERMINAL_STATES | {ExperimentState.PAUSED}


def _search_failure_detail(exc: Exception, experiment_uri: str) -> str:
    _LOGGER.warning(
        "search experiment failed for %s",
        experiment_uri,
        exc_info=exc,
    )
    if isinstance(exc, SearchError):
        return str(exc)
    if isinstance(exc, StorageError):
        return (
            "The search stopped because required artifact or state data is "
            "unavailable. Check the state directory and artifact URIs, then inspect "
            "the experiment."
        )
    if isinstance(exc, PluginRegistryError):
        return (
            "The search stopped because a required plugin is unavailable. Call "
            "math.find, reload the current plugin version, and start a "
            "new search."
        )
    return (
        "The search stopped because an artifact or plugin response was invalid. "
        "Check the reference contract and inspect the local Jacobian log before "
        "starting a new search."
    )


def _now() -> datetime:
    return datetime.now(UTC)


def _digest(payload: Any) -> str:
    return "sha256:" + hashlib.sha256(canonicalize_json(payload)).hexdigest()


def _environment_digest() -> str:
    return _digest(
        {
            "environment_version": "1",
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        }
    )


def _event_digest(event: SearchLifecycleEvent) -> str:
    payload = event.model_dump(mode="json")
    payload.pop("event_digest")
    return _digest(payload)


def _updated_snapshot(
    snapshot: SearchExperimentSnapshot,
    **changes: Any,
) -> SearchExperimentSnapshot:
    payload = snapshot.model_dump(mode="json")
    payload.update(changes)
    return SearchExperimentSnapshot.model_validate(payload)


def _updated_accounting(
    accounting: SearchAccounting,
    **changes: Any,
) -> SearchAccounting:
    payload = accounting.model_dump(mode="json")
    payload.update(changes)
    return SearchAccounting.model_validate(payload)


def _used_wall_ms(
    accounting: SearchAccounting,
    started: float,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    return accounting.wall_time_ms + math.ceil((clock() - started) * 1000)


def _is_verified_counterexample(result: ResultEnvelope) -> bool:
    return (
        result.assurance.verification is Verification.VERIFIED
        and result.conclusion is Conclusion.FALSE
    )


def _require_remaining_seconds(
    budget: SearchBudget,
    accounting: SearchAccounting,
    started: float,
    clock: Callable[[], float] = time.monotonic,
) -> float:
    remaining = budget.wall_seconds - _used_wall_ms(accounting, started, clock) / 1000
    if remaining < 1:
        raise _SearchBudgetExhaustedError("search wall-clock budget exhausted")
    return remaining


def _deduplicate_nominations(
    nominations: tuple[SearchNomination, ...],
) -> tuple[SearchNomination, ...]:
    selected: dict[str, SearchNomination] = {}
    for nomination in nominations:
        selected.setdefault(nomination.candidate_uri, nomination)
    return tuple(selected.values())


def _record_parents(
    records: list[SearchCandidateRecord],
    nominations: tuple[SearchNomination, ...],
) -> tuple[str, ...]:
    parents: dict[str, None] = {}
    for record in records:
        parents[record.candidate_uri] = None
        parents[record.evaluation_uri] = None
        if record.witness_uri is not None:
            parents[record.witness_uri] = None
        if record.verification_record_uri is not None:
            parents[record.verification_record_uri] = None
    for nomination in nominations:
        parents[nomination.candidate_uri] = None
    return tuple(parents)
