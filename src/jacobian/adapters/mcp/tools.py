"""MCP tool handlers for the capability surface."""

from __future__ import annotations

import logging
from typing import Annotated, Any, Literal

from mcp.server.mcpserver import Context
from pydantic import Field, StrictInt

from jacobian.adapters.mcp.constants import _CAPABILITY_SCOPE_RULE
from jacobian.adapters.mcp.context import AppState, _runtime
from jacobian.adapters.mcp.projections import (
    _capability_descriptor_view,
    _capability_discovery_response,
    _capability_inspection_extensions,
)
from jacobian.adapters.mcp.tooling import (
    AgentRecoveryError,
    _invoke_capability_attempt,
)
from jacobian.contracts.capabilities import (
    CapabilityInputKind,
    CapabilityMode,
    CapabilityResult,
)

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
                "Maximum compact discovery matches; defaults to 5. Start with "
                "5 and inspect only the strongest one or two candidates."
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
                "invocation examples; request it before invoking. FULL returns "
                "the complete installed descriptor for audit or client generation. "
                "Omit for discovery."
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
            "domain, mode, input_kind, artifact_type, limit, or cursor. Use one "
            "discovery call followed by "
            "one exact description call."
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
            "Call capability.describe with a mathematical query to search "
            "installed capabilities."
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
        response["invocations"] = [
            {
                "name": example.name,
                **(
                    {
                        "description": example.description,
                    }
                    if view == "FULL"
                    else {}
                ),
                "tool": "capability.invoke",
                "arguments": {
                    "capability_id": descriptor.capability_id,
                    "mode": example.mode.value,
                    "payload": example.input,
                },
            }
            for example in descriptor.invocation_examples
        ]
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
