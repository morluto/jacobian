"""MCP tool handlers for the capability surface."""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from mcp.server.mcpserver import Context
from mcp_types import CallToolResult, ResourceLink, TextContent
from pydantic import BaseModel, ConfigDict, Field, RootModel, StrictInt

from jacobian.adapters.mcp.constants import (
    _CAPABILITY_SCOPE_RULE,
    CAPABILITY_INSPECTION_RELATIONSHIPS_BYTE_LIMIT,
)
from jacobian.adapters.mcp.context import AppState, _runtime
from jacobian.adapters.mcp.projections import (
    _capability_descriptor_view,
    _capability_discovery_response,
    _capability_inspection_extensions,
    _compact_inspection_relationships,
)
from jacobian.adapters.mcp.tooling import (
    _invoke_capability_attempt,
)
from jacobian.contracts.capabilities import (
    CapabilityCatalogRelationshipKind,
    CapabilityDescriptor,
    CapabilityDiscoveryInspectCatalogRecoveryPath,
    CapabilityDiscoveryMatch,
    CapabilityDiscoveryRecoveryPath,
    CapabilityDiscoveryRequest,
    CapabilityId,
    CapabilityInputKind,
    CapabilityMode,
    CapabilityProviderAvailability,
    CapabilityResult,
)
from jacobian.contracts.common import ArtifactUri, CheckerUri, Sha256Digest
from jacobian.runtime.model import JacobianRuntime

CapabilityDescriptionView = Literal["SUMMARY", "CONTRACT", "FULL"]


class _MCPOutputModel(BaseModel):
    """Closed validation model for structured MCP output."""

    model_config = ConfigDict(extra="forbid")


class _CapabilityDiscoveryFields(_MCPOutputModel):
    """Shared catalog metadata with a closed MCP output shape."""

    policy_profile: str
    policy_digest: str


class _SchemaSummary(_MCPOutputModel):
    """Bounded facts extracted from a capability JSON Schema."""

    model_config = ConfigDict(populate_by_name=True)

    type: str | None
    required: tuple[str, ...]
    property_names: tuple[str, ...]
    ref: str | None = Field(default=None, alias="$ref")
    one_of_variants: StrictInt | None = None
    any_of_variants: StrictInt | None = None


class _RelatedCapability(_MCPOutputModel):
    capability_id: CapabilityId
    kind: CapabilityCatalogRelationshipKind | None = None
    relationship: str


class _DiscoveryInvocationExample(_MCPOutputModel):
    mode: CapabilityMode
    payload: dict[str, Any]


class _CapabilityDiscoveryOperationCard(CapabilityDiscoveryMatch):
    accepted_input_kinds: tuple[CapabilityInputKind, ...]
    accepted_artifact_types: tuple[ArtifactUri, ...]
    produced_artifact_types: tuple[ArtifactUri, ...]
    output_schema_summary: _SchemaSummary
    input_schema_summary: _SchemaSummary | None = None
    scope: Literal["EXACT_SUPPLIED_INPUT_OR_CLAIM"]
    assurance_ceiling: Literal["COMPUTED", "VERIFIED"]
    provider_availability: CapabilityProviderAvailability | Literal["UNKNOWN"]
    related_capabilities: tuple[_RelatedCapability, ...]
    invocation_example: _DiscoveryInvocationExample | None = None


class _ProviderRuntimeProjection(_MCPOutputModel):
    availability: CapabilityProviderAvailability
    version: str | None = None
    digest: Sha256Digest | None = None
    checker_ids: tuple[CheckerUri, ...] = ()
    diagnostic: str | None = None


class _CapabilityDescriptorProjection(_MCPOutputModel):
    """Typed SUMMARY/CONTRACT projection used only at the MCP boundary."""

    capability_id: CapabilityId
    version: str
    title: str
    description: str
    provider: str
    provider_runtime: _ProviderRuntimeProjection | None
    modes: tuple[CapabilityMode, ...]
    tags: tuple[str, ...] | None = None
    accepted_input_kinds: tuple[CapabilityInputKind, ...]
    accepted_artifact_types: tuple[ArtifactUri, ...]
    produced_artifact_types: tuple[ArtifactUri, ...]
    input_schema_summary: _SchemaSummary | None = None
    input_schema: dict[str, Any] | None = None
    output_schema_summary: _SchemaSummary
    has_invocation_examples: bool | None = None


class _CapabilityScopeRule(_MCPOutputModel):
    conclusion_scope: Literal["Only the exact supplied input or claim is covered."]
    bounded_repetition: str


class _CapabilityInvocationArguments(_MCPOutputModel):
    capability_id: CapabilityId
    mode: CapabilityMode
    payload: dict[str, Any]


class _CapabilityInvocation(_MCPOutputModel):
    name: str
    description: str | None = None
    tool: Literal["math.run"]
    arguments: _CapabilityInvocationArguments


class _SynchronousExecution(_MCPOutputModel):
    remote_safe_wall_seconds_max: StrictInt
    timeout_is_a_non_conclusion: bool
    larger_search_requires_multiple_bounded_invocations: bool
    backend_suitability: str


class _NextCapabilityViews(_MCPOutputModel):
    CONTRACT: str
    FULL: str


class _LeanWarmupHealth(_MCPOutputModel):
    status: str
    detail: str | None


class _LeanCacheDescription(_MCPOutputModel):
    key: str
    max_entries: StrictInt
    warmup_environment_variable: Literal["JACOBIAN_LEAN_WARMUP=1"]
    mathlib_warmup: _LeanWarmupHealth


class _CapabilitySearchArguments(_MCPOutputModel):
    query: str
    limit: Literal[5] = 5


class _CapabilitySearchRecoveryPath(_MCPOutputModel):
    action: Literal["search"]
    tool: Literal["math.find"] = "math.find"
    arguments: _CapabilitySearchArguments


CapabilityErrorRecoveryPath = (
    _CapabilitySearchRecoveryPath | CapabilityDiscoveryInspectCatalogRecoveryPath
)


class _CapabilityDiscoveryErrorDetail(_MCPOutputModel):
    code: Literal["INVALID_CURSOR", "UNKNOWN_CAPABILITY"]
    stage: Literal["capability_discovery", "capability_resolution"]
    message: str
    hint: str
    nearby_capability_ids: tuple[CapabilityId, ...] = ()
    available_recovery_paths: tuple[CapabilityErrorRecoveryPath, ...] = ()


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
    matches: tuple[_CapabilityDiscoveryOperationCard, ...]
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
    available_recovery_paths: tuple[CapabilityDiscoveryRecoveryPath, ...]
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
    capability: _CapabilityDescriptorProjection | CapabilityDescriptor
    scope_rule: _CapabilityScopeRule
    related_capabilities_byte_limit: StrictInt
    truncation_reason: str | None = None
    related_capabilities_truncated: bool
    invocations: tuple[_CapabilityInvocation, ...] | None = None
    related_capabilities: tuple[_RelatedCapability, ...] | None = None
    synchronous_execution: _SynchronousExecution | None = None
    next_views: _NextCapabilityViews | None = None
    cache: _LeanCacheDescription | None = None


class _CapabilityDiscoveryError(_MCPOutputModel):
    kind: Literal["error"]
    error: _CapabilityDiscoveryErrorDetail


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
    artifact_uris: tuple[str, ...] = (),
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
            ),
            *(
                ResourceLink(
                    uri=artifact_uri,
                    name=artifact_uri,
                    description="Durable Jacobian artifact returned by math.run.",
                    mime_type="application/json",
                )
                for artifact_uri in artifact_uris
            ),
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
        structured.model_dump(
            mode="json",
            by_alias=True,
            exclude_unset=True,
        ),
        _find_text_projection(response),
    )


def _unknown_capability_context(
    runtime: JacobianRuntime,
    capability_id: CapabilityId,
) -> dict[str, Any]:
    """Return bounded SDK-facing recovery without embedding the full catalog."""

    discovered = runtime.core.capabilities.discover(
        CapabilityDiscoveryRequest(query=capability_id, limit=5)
    )
    return {
        "nearby_capability_ids": [match.capability_id for match in discovered.matches],
        "available_recovery_paths": [
            {
                "action": "search",
                "tool": "math.find",
                "arguments": {"query": capability_id, "limit": 5},
            },
            {
                "action": "inspect_catalog",
                "resource_uri": "capability://catalog",
            },
        ],
    }


def _bounded_run_result(
    runtime: JacobianRuntime,
    result: CapabilityResult,
) -> CapabilityResult:
    """Keep unknown-capability recovery small in structured MCP output."""

    if not (result.diagnostics and result.diagnostics[0].code == "UNKNOWN_CAPABILITY"):
        return result
    payload = result.model_dump(mode="json")
    output = {
        key: value
        for key, value in payload["output"].items()
        if key != "available_capability_ids"
    }
    output.update(_unknown_capability_context(runtime, result.capability_id))
    payload["output"] = output
    return CapabilityResult.model_validate(payload)


def _run_text_projection(result: CapabilityResult) -> dict[str, Any]:
    payload = result.model_dump(mode="json")
    return {
        key: payload[key]
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
        CapabilityId | None,
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
    *,
    ctx: Context[AppState, Any],
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
            raise ValueError(
                "capability_id is an exact lookup and cannot be combined with query, "
                "domain, mode, input_kind, artifact_type, limit, or cursor. Use either "
                "discovery arguments or one exact capability_id in this call."
            )
        if capability_id is None:
            if view != "SUMMARY":
                raise ValueError(
                    "view applies only to an exact capability_id inspection; omit it "
                    "for search or browse."
                )
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
                    **_unknown_capability_context(
                        active_runtime,
                        capability_id,
                    ),
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
    capability_id: CapabilityId,
    payload: dict[str, Any],
    mode: CapabilityMode = CapabilityMode.EXPLORE,
    *,
    ctx: Context[AppState, Any],
) -> CapabilityRunToolResult:
    with _runtime(ctx) as active_runtime:
        result = await _invoke_capability_attempt(
            active_runtime,
            capability_id=capability_id,
            payload=payload,
            mode=mode,
            ctx=ctx,
        )
        result = _bounded_run_result(active_runtime, result)
        return _text_result(
            result.model_dump(mode="json"),
            _run_text_projection(result),
            result.artifact_uris,
        )
