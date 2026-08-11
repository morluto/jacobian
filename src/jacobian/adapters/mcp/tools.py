"""MCP tool handlers for the capability surface."""

from __future__ import annotations

import json
import logging
from typing import Annotated, Any, Literal

from mcp.server.mcpserver import Context
from mcp_types import CallToolResult, TextContent
from pydantic import BaseModel, ConfigDict, Field, RootModel, StrictInt

from jacobian.adapters.mcp.constants import (
    _CAPABILITY_SCOPE_RULE,
    CAPABILITY_INSPECTION_RELATIONSHIPS_BYTE_LIMIT,
    ReasoningLogMode,
)
from jacobian.adapters.mcp.context import AppState, _runtime
from jacobian.adapters.mcp.projections import (
    _capability_descriptor_view,
    _capability_discovery_response,
    _capability_inspection_extensions,
    _compact_inspection_relationships,
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


class _CapabilityDiscoveryFields(BaseModel):
    """Shared catalog metadata with a closed MCP output shape."""

    model_config = ConfigDict(extra="forbid")

    policy_profile: str
    policy_digest: str


class _CapabilityDiscoveryResult(_CapabilityDiscoveryFields):
    kind: Literal["discovery"]
    catalog_version: str
    catalog_digest: str
    discovery_version: Literal["1"]
    query: str | None = None
    domain: str | None = None
    domain_filter_status: Literal["UNFILTERED", "MATCHED", "UNKNOWN"]
    domain_filter_basis: str
    mode: CapabilityMode | None = None
    resolved_input_kind: CapabilityInputKind | None = None
    artifact_type: str | None = None
    routing_status: Literal["UNFILTERED", "ROUTES_FOUND", "NO_ROUTE"]
    routing_basis: str
    matches: list[dict[str, Any]]
    total_matches: StrictInt
    truncated: bool
    next_cursor: str | None = None
    available_domains: list[str]
    portfolio_fit: Literal[
        "UNFILTERED",
        "STRONG_CANDIDATES_FOUND",
        "ONLY_WEAK_LEXICAL_MATCHES",
        "NO_LEXICAL_MATCHES",
    ]
    portfolio_fit_basis: str
    available_recovery_paths: list[dict[str, Any]]
    recovery_paths_are_unranked: bool
    response_byte_limit: StrictInt
    truncation_reason: str | None = None
    available_domains_total: StrictInt
    available_domains_truncated: bool
    related_capabilities_truncated: bool
    match_metadata_truncated: bool


class _CapabilityInspectionResult(_CapabilityDiscoveryFields):
    kind: Literal["capability"]
    view: CapabilityDescriptionView
    capability: dict[str, Any]
    scope_rule: str | dict[str, Any]
    related_capabilities_byte_limit: StrictInt
    truncation_reason: str | None = None
    related_capabilities_truncated: bool
    invocations: list[dict[str, Any]] | None = None
    related_capabilities: list[dict[str, Any]] | None = None
    synchronous_execution: dict[str, Any] | None = None
    next_views: dict[str, str] | None = None
    cache: dict[str, Any] | None = None


class _CapabilityDiscoveryError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["error"]
    error: dict[str, Any]


class CapabilityDiscoveryResponse(
    RootModel[
        Annotated[
            _CapabilityDiscoveryResult
            | _CapabilityInspectionResult
            | _CapabilityDiscoveryError,
            Field(discriminator="kind"),
        ]
    ]
):
    """Closed, discriminated structured output for math.find."""

    # MCP tool output schemas describe structured content objects. Pydantic's
    # RootModel preserves the discriminated union but omits the common object
    # root from JSON Schema, which stricter MCP clients reject during tools/list.
    model_config = ConfigDict(json_schema_extra={"type": "object"})


CapabilityDiscoveryToolResult = Annotated[
    CallToolResult,
    CapabilityDiscoveryResponse,
]
CapabilityRunToolResult = Annotated[CallToolResult, CapabilityResult]


def _text_result(
    structured_content: dict[str, Any],
    text_projection: dict[str, Any],
) -> CallToolResult:
    """Keep one typed wire result plus a small agent-facing text view."""

    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=json.dumps(
                    text_projection,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        ],
        structured_content=structured_content,
    )


def _find_text_projection(response: dict[str, Any]) -> dict[str, Any]:
    if "error" in response:
        error = response["error"]
        return {
            "error": {
                key: error[key]
                for key in ("code", "stage", "message", "hint")
                if key in error
            }
        }
    if response.get("kind") == "discovery":
        return {
            "kind": "discovery",
            "matches": [
                {
                    key: match[key]
                    for key in (
                        "capability_id",
                        "title",
                        "description",
                        "modes",
                        "accepted_input_kinds",
                        "accepted_artifact_types",
                        "produced_artifact_types",
                        "input_schema_summary",
                        "output_schema_summary",
                        "scope",
                        "assurance_ceiling",
                        "provider_availability",
                        "related_capabilities",
                        "invocation_example",
                        "lexical_fit",
                    )
                    if key in match
                }
                for match in response.get("matches", [])
            ],
            "portfolio_fit": response.get("portfolio_fit"),
            "portfolio_fit_basis": response.get("portfolio_fit_basis"),
            "domain_filter_status": response.get("domain_filter_status"),
            "domain_filter_basis": response.get("domain_filter_basis"),
            "routing_status": response.get("routing_status"),
            "routing_basis": response.get("routing_basis"),
            "total_matches": response.get("total_matches"),
            "truncated": response.get("truncated"),
            "truncation_reason": response.get("truncation_reason"),
            "related_capabilities_truncated": response.get(
                "related_capabilities_truncated"
            ),
            "next_cursor": response.get("next_cursor"),
            "available_recovery_paths": response.get("available_recovery_paths"),
        }
    capability = response.get("capability", {})
    capability_projection: dict[str, Any] = {
        key: capability[key]
        for key in (
            "capability_id",
            "title",
            "description",
            "modes",
            "accepted_input_kinds",
            "accepted_artifact_types",
            "produced_artifact_types",
            "input_schema_summary",
            "output_schema_summary",
        )
        if key in capability
    }
    if response.get("view") == "CONTRACT" and "input_schema" in capability:
        capability_projection["input_schema"] = capability["input_schema"]
    projection: dict[str, Any] = {
        "kind": response.get("kind"),
        "view": response.get("view"),
        "capability": capability_projection,
        "scope_rule": response.get("scope_rule"),
    }
    if response.get("invocations"):
        projection["invocations"] = response["invocations"]
    if response.get("related_capabilities"):
        projection["related_capabilities"] = response["related_capabilities"]
    projection["related_capabilities_byte_limit"] = response.get(
        "related_capabilities_byte_limit"
    )
    projection["truncation_reason"] = response.get("truncation_reason")
    projection["related_capabilities_truncated"] = response.get(
        "related_capabilities_truncated"
    )
    return projection


def _find_result(response: dict[str, Any]) -> CallToolResult:
    if "error" in response and response.get("kind") != "error":
        response = {"kind": "error", **response}
    structured = CapabilityDiscoveryResponse.model_validate(response)
    return _text_result(
        structured.model_dump(mode="json", exclude_unset=True),
        _find_text_projection(response),
    )


def _run_text_projection(result: CapabilityResult) -> dict[str, Any]:
    payload = result.model_dump(mode="json")
    output = payload["output"]
    if (
        result.diagnostics
        and result.diagnostics[0].code == "UNKNOWN_CAPABILITY"
        and "available_capability_ids" in output
    ):
        output = {
            key: value
            for key, value in output.items()
            if key != "available_capability_ids"
        }
    return {
        key: output if key == "output" else payload[key]
        for key in (
            "capability_id",
            "mode",
            "execution",
            "output",
            "scope",
            "completeness",
            "relationships",
            "obligations",
            "assurance",
            "diagnostics",
            "artifact_uris",
        )
    }


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
            description=(
                "Plain-language mathematical outcome to find, such as computing an "
                "exact matrix determinant; no capability ID is required."
            ),
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
                "Maximum compact discovery matches; defaults to 5. Lower values "
                "reduce returned model context without changing match order."
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
                "Exact lookup only: SUMMARY judges fit; CONTRACT adds the validated "
                "input schema and invocation examples; FULL adds audit metadata. "
                "Omit for discovery."
            )
        ),
    ] = "SUMMARY",
    ctx: Context[AppState, Any] | None = None,
) -> CapabilityDiscoveryToolResult:
    with _runtime(ctx) as active_runtime:
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
            discovery_response = _capability_discovery_response(
                active_runtime,
                query=query,
                domain=domain,
                mode=mode,
                input_kind=input_kind,
                artifact_type=artifact_type,
                limit=limit,
                cursor=cursor,
            )
            return _find_result(discovery_response)
        capability_catalog = active_runtime.core.capabilities.catalog()
        descriptors = {
            item.capability_id: item for item in capability_catalog.capabilities
        }
        try:
            descriptor = descriptors[capability_id]
        except KeyError:
            hint = "Call math.find with a mathematical query to search installed capabilities."
            error_response = {
                "error": {
                    "code": "UNKNOWN_CAPABILITY",
                    "stage": "capability_resolution",
                    "message": f"Unknown capability: {capability_id}",
                    "hint": hint,
                    "available_capability_ids": sorted(descriptors),
                }
            }
            return _find_result(error_response)
        response: dict[str, Any] = {
            "kind": "capability",
            "view": view,
            "policy_profile": capability_catalog.policy_profile,
            "policy_digest": capability_catalog.policy_digest,
            "capability": _capability_descriptor_view(descriptor, view=view),
            "scope_rule": _CAPABILITY_SCOPE_RULE,
            "related_capabilities_byte_limit": (
                CAPABILITY_INSPECTION_RELATIONSHIPS_BYTE_LIMIT
            ),
            "truncation_reason": None,
            "related_capabilities_truncated": False,
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
            response.update(
                _capability_inspection_extensions(capability_id, descriptors)
            )
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
        _compact_inspection_relationships(response)
        return _find_result(response)


async def capability_invoke(
    capability_id: str,
    payload: dict[str, Any],
    mode: CapabilityMode = CapabilityMode.EXPLORE,
    ctx: Context[AppState, Any] | None = None,
) -> CapabilityRunToolResult:
    with _runtime(ctx) as active_runtime:
        result = await _invoke_capability_attempt(
            active_runtime,
            capability_id=capability_id,
            payload=payload,
            mode=mode,
            ctx=ctx,
        )
        return _text_result(
            result.model_dump(mode="json"), _run_text_projection(result)
        )


async def capability_invoke_reasoned(
    capability_id: str,
    payload: dict[str, Any],
    reasoning_run_id: ReasoningRunId,
    reasoning_call_id: ReasoningCallId,
    mode: CapabilityMode = CapabilityMode.EXPLORE,
    ctx: Context[AppState, Any] | None = None,
) -> CapabilityRunToolResult:
    with _runtime(ctx) as active_runtime:
        result = await _invoke_capability_attempt(
            active_runtime,
            capability_id=capability_id,
            payload=payload,
            mode=mode,
            ctx=ctx,
            reasoning_run_id=reasoning_run_id,
            reasoning_call_id=reasoning_call_id,
            reasoning_required=True,
        )
        return _text_result(
            result.model_dump(mode="json"), _run_text_projection(result)
        )


async def capability_invoke_audit(
    capability_id: str,
    payload: dict[str, Any],
    reasoning_run_id: ReasoningRunId | None = None,
    reasoning_call_id: ReasoningCallId | None = None,
    mode: CapabilityMode = CapabilityMode.EXPLORE,
    ctx: Context[AppState, Any] | None = None,
) -> CapabilityRunToolResult:
    with _runtime(ctx) as active_runtime:
        result = await _invoke_capability_attempt(
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
        return _text_result(
            result.model_dump(mode="json"), _run_text_projection(result)
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
    with _runtime(ctx) as active_runtime:
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
