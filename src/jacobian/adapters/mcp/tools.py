"""MCP tool handlers for the operation surface."""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from mcp.server.mcpserver import Context
from mcp_types import CallToolResult, ResourceLink, TextContent
from pydantic import BaseModel, ConfigDict, Field, RootModel, StrictInt

from jacobian.adapters.mcp.context import AppState, _catalog, _runtime
from jacobian.adapters.mcp.projections import (
    _operation_discovery_response,
)
from jacobian.adapters.mcp.tooling import (
    _invoke_operation_attempt,
)
from jacobian.contracts.common import ArtifactUri, ValueUri
from jacobian.contracts.operations import (
    OperationDescriptor,
    OperationDiscoveryMatch,
    OperationDiscoveryRequest,
    OperationId,
    OperationInputKind,
    OperationResult,
    OperationValuePort,
    ProviderAvailability,
)
from jacobian.runtime.model import JacobianRuntime


class _MCPOutputModel(BaseModel):
    """Closed validation model for structured MCP output."""

    model_config = ConfigDict(extra="forbid")


class _ValueReferenceArgument(_MCPOutputModel):
    value_ref: ValueUri


class _OperationDiscoveryOperationCard(OperationDiscoveryMatch):
    accepted_input_kinds: tuple[OperationInputKind, ...]
    accepted_artifact_types: tuple[ArtifactUri, ...]
    produced_artifact_types: tuple[ArtifactUri, ...]
    input_ports: tuple[OperationValuePort, ...]
    output_ports: tuple[OperationValuePort, ...]
    provider_availability: ProviderAvailability | Literal["UNKNOWN"]


class _OperationSearchRequest(_MCPOutputModel):
    op: Literal["search"]
    query: Annotated[str, Field(min_length=1, max_length=512)]
    domain: Annotated[
        str | None,
        Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,127}$"),
    ] = None
    input_kind: OperationInputKind | None = None
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


class _OperationInspectRequest(_MCPOutputModel):
    op: Literal["inspect"]
    operation_id: OperationId


OperationFindRequest = Annotated[
    _OperationSearchRequest | _OperationInspectRequest,
    Field(discriminator="op"),
]


class _OperationFindCallArguments(_MCPOutputModel):
    request: _OperationSearchRequest


class _OperationSearchRecoveryPath(_MCPOutputModel):
    action: Literal["search"]
    tool: Literal["math.find"] = "math.find"
    arguments: _OperationFindCallArguments


class _OperationCatalogPointer(_MCPOutputModel):
    action: Literal["inspect_catalog"]
    resource_uri: Literal["operation://catalog"] = "operation://catalog"


OperationErrorRecoveryPath = _OperationSearchRecoveryPath | _OperationCatalogPointer


class _OperationDiscoveryErrorDetail(_MCPOutputModel):
    code: Literal["INVALID_CURSOR", "UNKNOWN_OPERATION"]
    stage: Literal["operation_discovery", "operation_resolution"]
    message: str
    hint: str
    nearby_operation_ids: tuple[OperationId, ...] = ()
    available_recovery_paths: tuple[OperationErrorRecoveryPath, ...] = ()


class _OperationDiscoveryResult(_MCPOutputModel):
    kind: Literal["discovery"]
    discovery_version: Literal["1"]
    query: str
    domain: str | None = None
    input_kind: OperationInputKind | None = None
    artifact_type: str | None = None
    matches: tuple[_OperationDiscoveryOperationCard, ...]
    total_matches: StrictInt
    truncated: bool
    next_cursor: str | None = None
    catalog_resource: Literal["operation://catalog"]
    response_byte_limit: StrictInt
    truncation_reason: str | None = None
    match_metadata_truncated: bool


class _OperationInspectionResult(_MCPOutputModel):
    kind: Literal["operation"]
    operation: OperationDescriptor


class _OperationDiscoveryError(_MCPOutputModel):
    kind: Literal["error"]
    error: _OperationDiscoveryErrorDetail


class OperationFindResponse(
    RootModel[
        Annotated[
            _OperationDiscoveryResult
            | _OperationInspectionResult
            | _OperationDiscoveryError,
            Field(discriminator="kind"),
        ]
    ]
):
    """Closed, discriminated structured output for math.find."""

    # MCP tool output schemas describe structured content objects. Pydantic's
    # RootModel preserves the discriminated union but omits the common object
    # root from JSON Schema, which stricter MCP clients reject during tools/list.
    model_config = ConfigDict(json_schema_extra={"type": "object"})


OperationRunToolResult = Annotated[CallToolResult, OperationResult]


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


def _find_result(response: dict[str, Any]) -> OperationFindResponse:
    if "error" in response and response.get("kind") != "error":
        response = {"kind": "error", **response}
    return OperationFindResponse.model_validate(response)


def _unknown_operation_context(
    runtime: Any,
    operation_id: OperationId,
) -> dict[str, Any]:
    """Return bounded SDK-facing recovery without embedding the full catalog."""

    operations = getattr(getattr(runtime, "core", None), "operations", runtime)
    discovered = operations.discover(
        OperationDiscoveryRequest(query=operation_id, limit=5)
    )
    return {
        "nearby_operation_ids": [match.operation_id for match in discovered.matches],
        "available_recovery_paths": [
            {
                "action": "search",
                "tool": "math.find",
                "arguments": {
                    "request": {
                        "op": "search",
                        "query": operation_id,
                        "limit": 5,
                    }
                },
            },
            {
                "action": "inspect_catalog",
                "resource_uri": "operation://catalog",
            },
        ],
    }


def _bounded_run_result(
    runtime: JacobianRuntime,
    result: OperationResult,
) -> OperationResult:
    """Keep unknown-operation recovery small in structured MCP output."""

    if not (result.diagnostics and result.diagnostics[0].code == "UNKNOWN_OPERATION"):
        return result
    payload = result.model_dump(mode="json")
    output = payload["output"]
    output.update(_unknown_operation_context(runtime, result.operation_id))
    payload["output"] = output
    return OperationResult.model_validate(payload)


def _run_text_projection(result: OperationResult) -> dict[str, Any]:
    """Agent-visible projection: mathematical value first, then status."""
    payload = result.model_dump(mode="json")
    projection: dict[str, Any] = {
        "operation_id": payload["operation_id"],
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


def math_find(
    request: OperationFindRequest,
    *,
    ctx: Context[AppState, Any],
) -> OperationFindResponse:
    active_catalog = _catalog(ctx)
    if isinstance(request, _OperationSearchRequest):
        discovery_response = _operation_discovery_response(
            active_catalog,
            query=request.query,
            domain=request.domain,
            input_kind=request.input_kind,
            artifact_type=request.artifact_type,
            limit=request.limit,
            cursor=request.cursor,
        )
        return _find_result(discovery_response)
    operation_id = request.operation_id
    descriptor = active_catalog.inspect(operation_id)
    if descriptor is None:
        hint = "Call math.find with a mathematical query to search installed operations."
        error_response = {
            "error": {
                "code": "UNKNOWN_OPERATION",
                "stage": "operation_resolution",
                "message": f"Unknown operation: {operation_id}",
                "hint": hint,
                **_unknown_operation_context(
                    active_catalog,
                    operation_id,
                ),
            }
        }
        return _find_result(error_response)
    response: dict[str, Any] = {
        "kind": "operation",
        "operation": descriptor.model_dump(mode="json"),
    }
    return _find_result(response)


async def math_run(
    operation_id: OperationId,
    payload: dict[str, Any],
    inputs: dict[str, _ValueReferenceArgument] | None = None,
    *,
    ctx: Context[AppState, Any],
) -> OperationRunToolResult:
    """Run one math tool. Role comes from the tool ID."""
    with _runtime(ctx) as active_runtime:
        result = await _invoke_operation_attempt(
            active_runtime,
            operation_id=operation_id,
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
