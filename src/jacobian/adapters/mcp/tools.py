"""MCP tool handlers for the capability surface."""

from __future__ import annotations

import logging
from typing import Annotated, Any, Literal

from mcp.server.mcpserver import Context
from pydantic import Field, StrictInt

from jacobian.adapters.mcp.constants import _CAPABILITY_SCOPE_RULE, ReasoningLogMode
from jacobian.adapters.mcp.context import AppState, _runtime
from jacobian.adapters.mcp.projections import (
    _capability_descriptor_view,
    _capability_discovery_response,
    _capability_inspection_extensions,
)
from jacobian.adapters.mcp.tooling import (
    AgentRecoveryError,
    _invoke_capability_attempt,
    _run_blocking,
)
from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityCompletenessStatus,
    CapabilityInputKind,
    CapabilityMode,
    CapabilityResult,
)
from jacobian.contracts.reasoning import (
    ReasoningCallId,
    ReasoningInterpretationStatus,
    ReasoningPhase,
    ReasoningRunId,
    ReasoningWriteRequest,
    ReasoningWriteResult,
)
from jacobian.contracts.results import ExecutionStatus

_LOGGER = logging.getLogger(__name__)
CapabilityDescriptionView = Literal["SUMMARY", "CONTRACT", "FULL"]


async def capability_describe(
    capability_id: Annotated[
        str | None,
        Field(
            description=(
                "Exact installed ID; cannot be combined with discovery filters."
            )
        ),
    ] = None,
    query: Annotated[
        str | None,
        Field(
            min_length=1,
            max_length=512,
            description=("Mathematical outcome to find; no capability ID is required."),
        ),
    ] = None,
    domain: Annotated[
        str | None,
        Field(
            pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,127}$",
            description=(
                "Optional domain tag filter, such as universal_algebra, graph, "
                "polynomial, or lean."
            ),
        ),
    ] = None,
    mode: Annotated[
        CapabilityMode | None,
        Field(description="Optional EXPLORE or VERIFY capability filter."),
    ] = None,
    input_kind: Annotated[
        CapabilityInputKind | None,
        Field(description=("Input boundary used to reject incompatible routes.")),
    ] = None,
    artifact_type: Annotated[
        str | None,
        Field(
            pattern=r"^artifact://sha256/[0-9a-f]{64}$",
            description=(
                "Exact schema_uri from the stored artifact manifest; requires "
                "TYPED_ARTIFACT."
            ),
        ),
    ] = None,
    limit: Annotated[
        StrictInt | None,
        Field(
            ge=1,
            le=20,
            description=(
                "Maximum compact discovery matches; defaults to 5. This bounds "
                "the result and does not rank the agent's research choices."
            ),
        ),
    ] = None,
    cursor: Annotated[
        str | None,
        Field(
            max_length=128,
            pattern=r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)+$",
            description=(
                "Opaque continuation ID from next_cursor. Reuse the same query, "
                "domain, mode, input kind, artifact type, and limit."
            ),
        ),
    ] = None,
    view: Annotated[
        CapabilityDescriptionView,
        Field(
            description=(
                "Exact-lookup projection. SUMMARY is the small agent-facing "
                "default for judging fit. CONTRACT adds the validation-equivalent "
                "input schema, runtime identity, related operations, and validated "
                "invocation examples for constructing an unfamiliar request. FULL "
                "returns the complete installed descriptor for audit or client "
                "generation. Omit for discovery."
            )
        ),
    ] = "SUMMARY",
    ctx: Context[AppState, Any] | None = None,
) -> dict[str, Any]:
    active_runtime = _runtime(ctx)
    search_arguments = (
        query,
        domain,
        mode,
        input_kind,
        artifact_type,
        limit,
        cursor,
    )
    if capability_id is not None and any(
        argument is not None for argument in search_arguments
    ):
        raise AgentRecoveryError(
            "capability_id is an exact lookup and cannot be combined with query, "
            "domain, mode, input_kind, artifact_type, limit, or cursor. Use either "
            "discovery arguments or one exact capability_id in this call."
        )
    if capability_id is None:
        return _capability_discovery_response(
            active_runtime,
            query=query,
            domain=domain,
            mode=mode,
            input_kind=input_kind,
            artifact_type=artifact_type,
            limit=limit,
            cursor=cursor,
        )
    capability_catalog = active_runtime.core.capabilities.catalog()
    descriptors = {item.capability_id: item for item in capability_catalog.capabilities}
    try:
        descriptor = descriptors[capability_id]
    except KeyError:
        hint = (
            "Call math.find with a mathematical query to search installed capabilities."
        )
        return {
            "error": {
                "code": "UNKNOWN_CAPABILITY",
                "stage": "capability_resolution",
                "message": f"Unknown capability: {capability_id}",
                "hint": hint,
                "available_capability_ids": sorted(descriptors),
            }
        }
    response: dict[str, Any] = {
        "kind": "capability",
        "view": view,
        "policy_profile": capability_catalog.policy_profile,
        "policy_digest": capability_catalog.policy_digest,
        "capability": _capability_descriptor_view(descriptor, view=view),
        "scope_rule": _CAPABILITY_SCOPE_RULE,
    }
    if view == "SUMMARY":
        response["next_views"] = {
            "CONTRACT": (
                "Request before invocation for the validation-equivalent input "
                "schema and validated examples."
            ),
            "FULL": (
                "Request only for complete output schema, provider configuration, "
                "licensing, or audit metadata."
            ),
        }
    else:
        reasoning_mode = _reasoning_mode_from_context(ctx)
        invocations = []
        for example in descriptor.invocation_examples:
            entry: dict[str, Any] = {
                "name": example.name,
                **(
                    {
                        "description": example.description,
                    }
                    if view == "FULL"
                    else {}
                ),
                "tool": "math.run",
                "arguments": {
                    "capability_id": descriptor.capability_id,
                    "mode": example.mode.value,
                    "payload": example.input,
                },
            }
            if reasoning_mode is ReasoningLogMode.REQUIRED:
                entry["requires_reasoning_ids"] = True
                entry["protocol_note"] = (
                    "In REQUIRED mode, math.run also requires "
                    "reasoning_run_id and reasoning_call_id. Create a "
                    "reasoning run with reasoning.write (PLAN), then call "
                    "reasoning.write (BEFORE_TOOL) to obtain the IDs before "
                    "invoking."
                )
            invocations.append(entry)
        response["invocations"] = invocations
        response.update(_capability_inspection_extensions(capability_id, descriptors))
    if (
        view != "SUMMARY"
        and capability_id == "lean.check"
        and active_runtime.portfolio.lean_checkers
    ):
        response["cache"] = {
            "key": "exact content-addressed certificate and active checker digest",
            "max_entries": 128,
            "warmup_environment_variable": "JACOBIAN_LEAN_WARMUP=1",
            "mathlib_warmup": (
                active_runtime.portfolio.lean.mathlib_warmup_health()
                if active_runtime.portfolio.lean is not None
                else {"status": "UNAVAILABLE", "detail": None}
            ),
        }
    return response


async def capability_invoke(
    capability_id: str,
    payload: dict[str, Any],
    mode: CapabilityMode = CapabilityMode.EXPLORE,
    ctx: Context[AppState, Any] | None = None,
) -> CapabilityResult:
    active_runtime = _runtime(ctx)
    return await _invoke_capability_attempt(
        active_runtime,
        capability_id=capability_id,
        payload=payload,
        mode=mode,
        ctx=ctx,
    )


async def capability_invoke_reasoned(
    capability_id: str,
    payload: dict[str, Any],
    reasoning_run_id: ReasoningRunId,
    reasoning_call_id: ReasoningCallId,
    mode: CapabilityMode = CapabilityMode.EXPLORE,
    ctx: Context[AppState, Any] | None = None,
) -> CapabilityResult:
    active_runtime = _runtime(ctx)
    return await _invoke_capability_attempt(
        active_runtime,
        capability_id=capability_id,
        payload=payload,
        mode=mode,
        ctx=ctx,
        reasoning_run_id=reasoning_run_id,
        reasoning_call_id=reasoning_call_id,
        reasoning_required=True,
    )


async def capability_invoke_audit(
    capability_id: str,
    payload: dict[str, Any],
    reasoning_run_id: ReasoningRunId | None = None,
    reasoning_call_id: ReasoningCallId | None = None,
    mode: CapabilityMode = CapabilityMode.EXPLORE,
    ctx: Context[AppState, Any] | None = None,
) -> CapabilityResult:
    active_runtime = _runtime(ctx)
    return await _invoke_capability_attempt(
        active_runtime,
        capability_id=capability_id,
        payload=payload,
        mode=mode,
        ctx=ctx,
        reasoning_run_id=reasoning_run_id,
        reasoning_call_id=reasoning_call_id,
        reasoning_required=False,
        reasoning_audit=True,
    )


async def reasoning_write(
    phase: ReasoningPhase,
    summary: Annotated[str, Field(min_length=1, max_length=512)],
    run_id: ReasoningRunId | None = None,
    call_id: ReasoningCallId | None = None,
    capability_id: str | None = None,
    mode: CapabilityMode | None = None,
    interpretation_status: ReasoningInterpretationStatus | None = None,
    reported_execution_status: ExecutionStatus | None = None,
    reported_assurance_level: CapabilityAssuranceLevel | None = None,
    reported_completeness_status: CapabilityCompletenessStatus | None = None,
    ctx: Context[AppState, Any] | None = None,
) -> ReasoningWriteResult:
    active_runtime = _runtime(ctx)
    request = ReasoningWriteRequest(
        phase=phase,
        summary=summary,
        run_id=run_id,
        call_id=call_id,
        capability_id=capability_id,
        mode=mode,
        interpretation_status=interpretation_status,
        reported_execution_status=reported_execution_status,
        reported_assurance_level=reported_assurance_level,
        reported_completeness_status=reported_completeness_status,
    )
    if phase is ReasoningPhase.BEFORE_TOOL:
        descriptors = {
            item.capability_id: item
            for item in active_runtime.core.capabilities.catalog().capabilities
        }
        descriptor = descriptors.get(str(capability_id))
        if descriptor is None:
            raise AgentRecoveryError(
                "BEFORE_TOOL names an unavailable capability. Use math.find "
                "to select an installed capability ID."
            )
        if mode not in descriptor.modes:
            raise AgentRecoveryError(
                "BEFORE_TOOL selects a mode the capability does not advertise. "
                "Inspect its CONTRACT view and choose an installed mode."
            )
    return await _run_blocking(active_runtime.core.reasoning_log.write, request)


def _reasoning_mode_from_context(
    ctx: Context[AppState, Any] | None,
) -> ReasoningLogMode:
    """Read the server's reasoning-log mode from the request lifespan state."""

    if ctx is None:
        return ReasoningLogMode.OFF
    state = ctx.request_context.lifespan_context
    if isinstance(state, AppState):
        return state.reasoning_log_mode
    return ReasoningLogMode.OFF
