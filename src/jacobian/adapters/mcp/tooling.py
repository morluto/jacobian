"""MCP tool invocation helpers: annotations, logging, and capability attempts."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from mcp_types import ToolAnnotations

from jacobian.adapters.mcp.projections import (
    _invoke_capability_with_cancellation,
)
from jacobian.canonical import canonicalize_json
from jacobian.contracts.capabilities import (
    CapabilityMode,
    CapabilityRequest,
    CapabilityResult,
)
from jacobian.reasoning_log import ReasoningProtocolError

_LOGGER = logging.getLogger(__name__)


async def _run_blocking[BlockingResultT](
    function: Callable[..., BlockingResultT],
    /,
    *args: Any,
    on_cancel: Callable[[], None] | None = None,
) -> BlockingResultT:
    """Run blocking MCP work without detaching it from request teardown.

    On cancellation, waits for the worker to drain and stores the result
    in the ``drained_result`` attribute so the caller can persist a
    completed outcome instead of unconditionally recording cancellation.
    """

    worker = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError as exc:
        if on_cancel is not None:
            on_cancel()
        drained: BlockingResultT | None = None
        try:
            drained = await asyncio.shield(worker)
        except Exception:
            _LOGGER.debug(
                "blocking MCP worker failed while its cancelled request drained",
                exc_info=True,
            )
        exc.drained_result = drained  # type: ignore[attr-defined]
        raise


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


def _request_trace_digest(ctx: Any | None) -> tuple[str, str]:
    """Return a bounded correlation digest without retaining caller identifiers."""

    if ctx is None:
        return "none", "none"
    headers = getattr(ctx, "headers", None)
    if headers is not None:
        traceparent = headers.get("traceparent")
        if isinstance(traceparent, str) and 0 < len(traceparent) <= 256:
            digest = hashlib.sha256(traceparent.encode("utf-8")).hexdigest()[:8]
            return digest, "traceparent"
    try:
        request_id = str(ctx.request_id)
    except (AttributeError, TypeError, ValueError):
        return "none", "none"
    if not request_id or len(request_id) > 256:
        return "none", "none"
    digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:8]
    return digest, "request_id"


def _response_size(value: Any) -> int:
    try:
        if hasattr(value, "model_dump_json"):
            return len(value.model_dump_json().encode("utf-8"))
        return len(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        )
    except (TypeError, ValueError):
        return -1


def _log_capability_attempt(
    *,
    capability_id: str,
    mode: CapabilityMode,
    started: float,
    argument_digest: str,
    trace_digest: str,
    trace_source: str,
    result: CapabilityResult | None = None,
    execution_status: str | None = None,
    diagnostic_codes: tuple[str, ...] = (),
) -> None:
    if result is not None:
        execution_status = result.execution.status.value
        diagnostic_codes = tuple(item.code for item in result.diagnostics)
        capability_version = result.capability_version
        assurance = result.assurance.level.value
        operation_runtime_ms = result.execution.runtime_ms
        response_bytes = _response_size(result)
    else:
        capability_version = "unknown"
        assurance = "none"
        operation_runtime_ms = None
        response_bytes = 0
    codes = ",".join(diagnostic_codes[:8]) or "none"
    _LOGGER.info(
        "MCP capability attempt trace_digest=%s trace_source=%s "
        "capability_id=%s capability_version=%s mode=%s "
        "execution_status=%s assurance=%s diagnostic_codes=%s "
        "attempt_duration_ms=%.3f operation_runtime_ms=%s "
        "response_bytes=%d argument_digest=%s",
        trace_digest,
        trace_source,
        capability_id,
        capability_version,
        mode.value,
        execution_status or "ERROR",
        assurance,
        codes,
        (time.monotonic() - started) * 1000,
        "none" if operation_runtime_ms is None else operation_runtime_ms,
        response_bytes,
        argument_digest,
    )


def _claim_reasoning_call(
    runtime: Any,
    *,
    run_id: str | None,
    call_id: str | None,
    capability_id: str,
    mode: CapabilityMode,
    argument_digest: str,
    required: bool,
    audit: bool,
) -> bool:
    if run_id is None or call_id is None:
        if required:
            raise ReasoningProtocolError(
                "REASONING_LOG_REQUIRED",
                "math.run requires the run_id and call_id from BEFORE_TOOL.",
                "Call reasoning.write with PLAN, then BEFORE_TOOL, and retry with its IDs.",
            )
        if audit:
            _LOGGER.warning(
                "MCP capability reasoning protocol violation; unbound invocation "
                "allowed in audit mode"
            )
        return False
    try:
        runtime.core.reasoning_log.claim_call(
            run_id,
            call_id,
            capability_id,
            mode,
            argument_digest,
        )
    except ReasoningProtocolError:
        if required:
            raise
        _LOGGER.warning(
            "MCP capability reasoning protocol violation; invoking in audit mode",
            exc_info=True,
        )
        return False
    return True


def _finish_failed_reasoning_call(
    runtime: Any,
    *,
    bound: bool,
    run_id: str | None,
    call_id: str | None,
    capability_id: str,
    mode: CapabilityMode,
    argument_digest: str,
    execution_status: str,
    diagnostic_code: str,
) -> None:
    if not bound:
        return
    assert run_id is not None
    assert call_id is not None
    runtime.core.reasoning_log.finish_call(
        run_id,
        call_id,
        capability_id,
        mode,
        argument_digest,
        execution_status=execution_status,
        diagnostic_codes=(diagnostic_code,),
    )


async def _invoke_capability_attempt(
    runtime: Any,
    *,
    capability_id: str,
    payload: dict[str, Any],
    mode: CapabilityMode,
    ctx: Any | None,
    reasoning_run_id: str | None = None,
    reasoning_call_id: str | None = None,
    reasoning_required: bool = False,
    reasoning_audit: bool = False,
) -> CapabilityResult:
    started = time.monotonic()
    argument_digest = _argument_digest(
        {
            "capability_id": capability_id,
            "mode": mode.value,
            "payload": payload,
        }
    )
    trace_digest, trace_source = _request_trace_digest(ctx)
    cancellation_event = threading.Event()
    bound = _claim_reasoning_call(
        runtime,
        run_id=reasoning_run_id,
        call_id=reasoning_call_id,
        capability_id=capability_id,
        mode=mode,
        argument_digest=argument_digest,
        required=reasoning_required,
        audit=reasoning_audit,
    )
    try:
        result = await _run_blocking(
            _invoke_capability_with_cancellation,
            runtime,
            CapabilityRequest(
                capability_id=capability_id,
                mode=mode,
                input=payload,
            ),
            cancellation_event,
            on_cancel=cancellation_event.set,
        )
    except asyncio.CancelledError as exc:
        drained = getattr(exc, "drained_result", None)
        if isinstance(drained, CapabilityResult):
            if bound:
                assert reasoning_run_id is not None
                assert reasoning_call_id is not None
                runtime.core.reasoning_log.finish_call(
                    reasoning_run_id,
                    reasoning_call_id,
                    capability_id,
                    mode,
                    argument_digest,
                    result=drained,
                )
            _log_capability_attempt(
                capability_id=capability_id,
                mode=mode,
                started=started,
                argument_digest=argument_digest,
                trace_digest=trace_digest,
                trace_source=trace_source,
                result=drained,
            )
        else:
            _finish_failed_reasoning_call(
                runtime,
                bound=bound,
                run_id=reasoning_run_id,
                call_id=reasoning_call_id,
                capability_id=capability_id,
                mode=mode,
                argument_digest=argument_digest,
                execution_status="CANCELLED",
                diagnostic_code="CLIENT_CANCELLED",
            )
            _log_capability_attempt(
                capability_id=capability_id,
                mode=mode,
                started=started,
                argument_digest=argument_digest,
                trace_digest=trace_digest,
                trace_source=trace_source,
                execution_status="CANCELLED",
                diagnostic_codes=("CLIENT_CANCELLED",),
            )
        raise
    except Exception:
        _finish_failed_reasoning_call(
            runtime,
            bound=bound,
            run_id=reasoning_run_id,
            call_id=reasoning_call_id,
            capability_id=capability_id,
            mode=mode,
            argument_digest=argument_digest,
            execution_status="ERROR",
            diagnostic_code="INVOCATION_EXCEPTION",
        )
        _log_capability_attempt(
            capability_id=capability_id,
            mode=mode,
            started=started,
            argument_digest=argument_digest,
            trace_digest=trace_digest,
            trace_source=trace_source,
            execution_status="ERROR",
            diagnostic_codes=("INVOCATION_EXCEPTION",),
        )
        raise
    if bound:
        assert reasoning_run_id is not None
        assert reasoning_call_id is not None
        runtime.core.reasoning_log.finish_call(
            reasoning_run_id,
            reasoning_call_id,
            capability_id,
            mode,
            argument_digest,
            result=result,
        )
    _log_capability_attempt(
        capability_id=capability_id,
        mode=mode,
        started=started,
        argument_digest=argument_digest,
        trace_digest=trace_digest,
        trace_source=trace_source,
        result=result,
    )
    return result
