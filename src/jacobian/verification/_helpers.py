"""Authorized witness and certificate replay services."""

from __future__ import annotations

import hashlib
import logging
import platform
from typing import Any

from pydantic import ValidationError

from jacobian.canonical import canonicalize_json
from jacobian.contracts.operations import ProviderObservation
from jacobian.registry import (
    CheckerRegistryError,
)
from jacobian.schema_registry import SchemaRegistryError
from jacobian.storage.errors import (
    ArtifactIntegrityError,
    StorageError,
)
from jacobian.verification.checker_protocol import CheckerWorkerFailure

_LOGGER = logging.getLogger(__name__)


_CHECKER_OUTPUT_TOO_LARGE = (
    "The checker returned too much data. Retry with a smaller input "
    "and inspect the local checker log if the limit is reached again."
)
_CHECKER_DIAGNOSTICS_TOO_LARGE = (
    "The checker produced too many diagnostics. Retry with a smaller input "
    "and inspect the local checker log if the limit is reached again."
)
_CHECKER_UNREADABLE_RESPONSE = (
    "The checker returned an unreadable response. Retry once; "
    "if it happens again, inspect the local checker log."
)
_CHECKER_CHANGED = (
    "The checker changed after authorization. Authorize the current checker version, "
    "then retry."
)
_CHECKER_STOPPED = (
    "The checker stopped before returning a decision. Retry once; "
    "if it happens again, inspect the local checker log."
)
_CHECKER_INVALID_DECISION = (
    "The checker returned an invalid decision. Inspect the local checker log "
    "before retrying."
)
_CHECKER_TIMEOUT = (
    "The checker did not finish within the allowed time. "
    "Retry with a smaller input and inspect the local checker log if it times out again."
)
_CHECKER_CANCELLED = "The checker was cancelled before returning a decision."


def _digest_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _environment_digest(
    implementation_digest: str,
    provider_runtime: ProviderObservation | None = None,
) -> str:
    identity: dict[str, Any] = {
        "environment_version": "1",
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "implementation_digest": implementation_digest,
    }
    if provider_runtime is not None:
        identity["provider_runtime"] = provider_runtime.model_dump(mode="json")
    return _digest_bytes(canonicalize_json(identity))


def _checker_failure_detail(response: CheckerWorkerFailure) -> str:
    match response.error_code:
        case "SOURCE_CHANGED":
            return _CHECKER_CHANGED
        case "RESPONSE_INVALID":
            return _CHECKER_INVALID_DECISION
        case (
            "EXECUTION_FAILED"
            | "INVALID_REQUEST"
            | "MALFORMED_RUNTIME"
            | "UNDECLARED_IMPORT"
        ):
            return _CHECKER_STOPPED


def _verification_input_failure_detail(exc: Exception) -> str:
    if isinstance(exc, CheckerRegistryError):
        return str(exc)
    if isinstance(exc, ValueError) and not isinstance(
        exc,
        (SchemaRegistryError, ValidationError),
    ):
        return str(exc)
    _LOGGER.warning("verification input validation failed", exc_info=exc)
    return (
        "The evidence does not match its registered schema. Recreate it from the "
        "reference contract and the exact claim and candidate, then retry."
    )


def _verification_storage_failure_detail(exc: StorageError) -> str:
    if isinstance(exc, ArtifactIntegrityError):
        return (
            "Jacobian detected corrupted local verification data. Restore the "
            "state directory from a trusted backup, then retry."
        )
    return (
        "Jacobian could not read or save verification data. Check the state "
        "directory and available disk space, then retry."
    )
