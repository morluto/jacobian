"""Persistent bounded-enumeration experiments."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from jacobian.contracts.discovery import (
    ExperimentSnapshot,
    ExperimentState,
)
from jacobian.experiments.errors import ExperimentError
from jacobian.plugins.registry import PluginRegistryError
from jacobian.storage.errors import StorageError

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


def _enumeration_failure_detail(exc: Exception, experiment_uri: str) -> str:
    _LOGGER.warning(
        "enumeration experiment failed for %s",
        experiment_uri,
        exc_info=exc,
    )
    if isinstance(exc, ExperimentError):
        return str(exc)
    if isinstance(exc, StorageError):
        return (
            "The enumeration stopped because required artifact or state data is "
            "unavailable. Check the state directory and artifact URIs, then inspect "
            "the experiment."
        )
    if isinstance(exc, PluginRegistryError):
        return (
            "The enumeration stopped because a required plugin is unavailable. "
            "Call math.find, reload the current plugin version, and start "
            "a new experiment."
        )
    return (
        "The enumeration stopped because an artifact or plugin response was "
        "invalid. Check the reference contract and inspect the local Jacobian log "
        "before starting a new experiment."
    )


def _now() -> datetime:
    return datetime.now(UTC)


def _updated(
    snapshot: ExperimentSnapshot,
    **changes: Any,
) -> ExperimentSnapshot:
    payload = snapshot.model_dump(mode="json")
    payload.update(changes)
    return ExperimentSnapshot.model_validate(payload)
