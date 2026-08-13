"""MCP tool handlers for the capability surface."""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from mcp.server.mcpserver import Context
from mcp_types import CallToolResult, ResourceLink, TextContent
from pydantic import BaseModel, ConfigDict, Field, RootModel, StrictInt

from jacobian.adapters.mcp.context import AppState, _runtime
from jacobian.adapters.mcp.projections import (
    _capability_discovery_response,
)
from jacobian.adapters.mcp.tooling import (
    _invoke_capability_attempt,
)
from jacobian.contracts.capabilities import (
    CapabilityDescriptor,
    CapabilityDiscoveryMatch,
    CapabilityDiscoveryRequest,
    CapabilityId,
    CapabilityInputKind,
    CapabilityProviderAvailability,
    CapabilityResult,
    CapabilityValuePort,
)
from jacobian.contracts.common import ArtifactUri, ValueUri
from jacobian.runtime.model import JacobianRuntime


class _MCPOutputModel(BaseModel):
    """Closed validation model for structured MCP output."""

    model_config = ConfigDict(extra="forbid")


class _ValueReferenceArgument(_MCPOutputModel):
    value_ref: ValueUri


class _CapabilityDiscoveryOperationCard(CapabilityDiscoveryMatch):
    accepted_input_kinds: tuple[CapabilityInputKind, ...]
    accepted_artifact_types: tuple[ArtifactUri, ...]
    produced_artifact_types: tuple[ArtifactUri, ...]
    input_ports: tuple[CapabilityValuePort, ...]
    output_ports: tuple[CapabilityValuePort, ...]
    provider_availability: CapabilityProviderAvailability | Literal["UNKNOWN"]


class _CapabilitySearchRequest(_MCPOutputModel):
    op: Literal["search"]
    query: Annotated[str, Field(min_length=1, max_length=512)]
    domain: Annotated[
        str | None,
        Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,127}$"),
    ] = None
    input_kind: CapabilityInputKind | None = None
    artifact_type: Annotated[
        str | None,
        Field(pattern=r"^artifact://sha256/[0-9a-f]{64}$"),
    ] = None
    limit: Annotated[StrictInt, Field(ge=1, le=20)] = 5
    cursor: Annotated[
        str | None,
        Field(
            max_length=128,
            pattern=r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)+$",
        ),
    ] = None


class _CapabilityInspectRequest(_MCPOutputModel):
    op: Literal["inspect"]
    capability_id: CapabilityId


CapabilityFindRequest = Annotated[
    _CapabilitySearchRequest | _CapabilityInspectRequest,
    Field(discriminator="op"),
]


class _CapabilityFindCallArguments(_MCPOutputModel):
    request: _CapabilitySearchRequest


class _CapabilitySearchRecoveryPath(_MCPOutputModel):
    action: Literal["search"]
    tool: Literal["math.find"] = "math.find"
    arguments: _CapabilityFindCallArguments


class _CapabilityCatalogPointer(_MCPOutputModel):
    action: Literal["inspect_catalog"]
    resource_uri: Literal["capability://catalog"] = "capability://catalog"


CapabilityErrorRecoveryPath = _CapabilitySearchRecoveryPath | _CapabilityCatalogPointer


class _CapabilityDiscoveryErrorDetail(_MCPOutputModel):
    code: Literal["INVALID_CURSOR", "UNKNOWN_CAPABILITY"]
    stage: Literal["capability_discovery", "capability_resolution"]
    message: str
    hint: str
    nearby_capability_ids: tuple[CapabilityId, ...] = ()
    available_recovery_paths: tuple[CapabilityErrorRecoveryPath, ...] = ()


class _CapabilityDiscoveryResult(_MCPOutputModel):
    kind: Literal["discovery"]
    discovery_version: Literal["1"]
    query: str
    domain: str | None = None
    input_kind: CapabilityInputKind | None = None
    artifact_type: str | None = None
    matches: tuple[_CapabilityDiscoveryOperationCard, ...]
    total_matches: StrictInt
    truncated: bool
    next_cursor: str | None = None
    catalog_resource: Literal["capability://catalog"]
    response_byte_limit: StrictInt
    truncation_reason: str | None = None
    match_metadata_truncated: bool


class _CapabilityInspectionResult(_MCPOutputModel):
    kind: Literal["capability"]
    capability: CapabilityDescriptor


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
                        "accepted_input_kinds",
                        "accepted_artifact_types",
                        "produced_artifact_types",
                        "input_ports",
                        "output_ports",
                        "provider_availability",
                        "relevance_score",
                        "applicability",
                        "applicability_code",
                    )
                    if key in match
                }
                for match in response.get("matches", [])
            ],
            "total_matches": response.get("total_matches"),
            "truncated": response.get("truncated"),
            "truncation_reason": response.get("truncation_reason"),
            "next_cursor": response.get("next_cursor"),
            "catalog_resource": response.get("catalog_resource"),
        }
    capability = response.get("capability", {})
    capability_projection: dict[str, Any] = {
        key: capability[key]
        for key in (
            "capability_id",
            "title",
            "description",
            "accepted_input_kinds",
            "accepted_artifact_types",
            "produced_artifact_types",
            "input_ports",
            "output_ports",
        )
        if key in capability
    }
    projection: dict[str, Any] = {
        "kind": response.get("kind"),
        "capability": capability_projection,
    }
    return projection


def _find_result(response: dict[str, Any]) -> CallToolResult:
    if "error" in response and response.get("kind") != "error":
        response = {"kind": "error", **response}
    return _text_result(
        response,
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
                "arguments": {
                    "request": {
                        "op": "search",
                        "query": capability_id,
                        "limit": 5,
                    }
                },
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
    output = payload["output"]
    output.update(_unknown_capability_context(runtime, result.capability_id))
    payload["output"] = output
    return CapabilityResult.model_validate(payload)


def _run_text_projection(result: CapabilityResult) -> dict[str, Any]:
    """Agent-visible projection: mathematical value first, then status."""
    payload = result.model_dump(mode="json")
    projection: dict[str, Any] = {
        "capability_id": payload["capability_id"],
        "output": payload["output"],
        "execution": payload["execution"],
    }
    for key in (
        "diagnostics",
        "verification_record_uri",
        "artifact_uris",
    ):
        value = payload.get(key)
        if value not in (None, [], (), {}):
            projection[key] = value
    return projection


async def math_find(
    request: CapabilityFindRequest,
    *,
    ctx: Context[AppState, Any],
) -> CapabilityDiscoveryToolResult:
    with _runtime(ctx) as active_runtime:
        if isinstance(request, _CapabilitySearchRequest):
            discovery_response = _capability_discovery_response(
                active_runtime,
                query=request.query,
                domain=request.domain,
                input_kind=request.input_kind,
                artifact_type=request.artifact_type,
                limit=request.limit,
                cursor=request.cursor,
            )
            return _find_result(discovery_response)
        capability_id = request.capability_id
        descriptor = active_runtime.core.capabilities.inspect(capability_id)
        if descriptor is None:
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
            "capability": descriptor.model_dump(mode="json"),
        }
        return _find_result(response)


async def math_run(
    capability_id: CapabilityId,
    payload: dict[str, Any],
    inputs: dict[str, _ValueReferenceArgument] | None = None,
    *,
    ctx: Context[AppState, Any],
) -> CapabilityRunToolResult:
    """Run one math tool. Role comes from the tool ID."""
    with _runtime(ctx) as active_runtime:
        result = await _invoke_capability_attempt(
            active_runtime,
            capability_id=capability_id,
            payload=payload,
            inputs=(
                {name: binding.value_ref for name, binding in inputs.items()}
                if inputs is not None
                else {}
            ),
            ctx=ctx,
        )
        result = _bounded_run_result(active_runtime, result)
        return _text_result(
            result.model_dump(mode="json"),
            _run_text_projection(result),
            result.artifact_uris,
        )
