"""MCP tool invocation helpers: annotations, logging, and operation attempts."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from mcp_types import ToolAnnotations

from jacobian.canonical import canonicalize_json
from jacobian.contracts.operations import (
    OperationRequest,
    OperationResult,
)

_LOGGER = logging.getLogger(__name__)


class AgentRecoveryError(RuntimeError):
    """A safe, actionable failure intended for an agent tool response."""


def _tool_annotations(
    *,
    read_only: bool = False,
    destructive: bool = False,
    idempotent: bool = False,
) -> ToolAnnotations:
    return ToolAnnotations(
        read_only_hint=read_only,
        destructive_hint=destructive,
        idempotent_hint=idempotent,
        open_world_hint=False,
    )


def _argument_digest(arguments: dict[str, Any]) -> str:
    try:
        encoded = canonicalize_json(arguments)
    except (TypeError, ValueError):
        encoded = json.dumps(
            arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _log_operation_attempt(
    *,
    operation_id: str,
    argument_digest: str,
    provider: str,
    checker_ids: tuple[str, ...],
    result: OperationResult | None = None,
    execution_status: str | None = None,
    diagnostic_codes: tuple[str, ...] = (),
) -> None:
    if result is not None:
        execution_status = result.execution.status.value
        diagnostic_codes = tuple(item.code for item in result.diagnostics)
        operation_version = result.operation_version
        verification_record_uri_present = result.verification_record_uri is not None
        artifact_count = len(result.artifact_uris)
    else:
        operation_version = "unknown"
        verification_record_uri_present = False
        artifact_count = 0
    codes = ",".join(diagnostic_codes[:8]) or "none"
    checkers = ",".join(checker_ids) or "none"
    _LOGGER.info(
        "MCP operation attempt argument_digest=%s operation_id=%s "
        "operation_version=%s provider=%s checker_ids=%s "
        "execution_status=%s verification_record_uri_present=%s diagnostic_codes=%s "
        "artifact_count=%d",
        argument_digest,
        operation_id,
        operation_version,
        provider,
        checkers,
        execution_status or "ERROR",
        verification_record_uri_present,
        codes,
        artifact_count,
    )


def _invoke_operation_attempt(
    runtime: Any,
    *,
    operation_id: str,
    payload: dict[str, Any],
    inputs: dict[str, str] | None = None,
) -> OperationResult:
    argument_digest = _argument_digest(
        {
            "operation_id": operation_id,
            "payload": payload,
            "inputs": inputs or {},
        }
    )
    descriptor = runtime.core.operations.inspect(operation_id)
    provider = descriptor.provider if descriptor is not None else "unknown"
    checker_ids = (
        tuple(str(checker_id) for checker_id in descriptor.provider_runtime.checker_ids)
        if descriptor is not None and descriptor.provider_runtime is not None
        else ()
    )
    request = OperationRequest(
        operation_id=operation_id,
        input=payload,
        inputs=inputs or {},
    )
    try:
        result: OperationResult = runtime.core.operations.invoke(request)
    except Exception:
        _log_operation_attempt(
            operation_id=operation_id,
            argument_digest=argument_digest,
            provider=provider,
            checker_ids=checker_ids,
            execution_status="ERROR",
            diagnostic_codes=("INVOCATION_EXCEPTION",),
        )
        raise
    _log_operation_attempt(
        operation_id=operation_id,
        argument_digest=argument_digest,
        provider=provider,
        checker_ids=checker_ids,
        result=result,
    )
    return result
