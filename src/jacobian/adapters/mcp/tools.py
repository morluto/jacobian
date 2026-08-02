"""MCP tool handlers for capability and workspace surfaces."""

from __future__ import annotations

import logging
from typing import Annotated, Any, Literal

from mcp.server.mcpserver import Context
from mcp_types import CallToolResult
from pydantic import Field, StrictInt

from jacobian.adapters.mcp.constants import _CAPABILITY_SCOPE_RULE
from jacobian.adapters.mcp.context import AppState, _projection_strategy, _runtime
from jacobian.adapters.mcp.projections import (
    _capability_call_tool_result,
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
    CapabilityInputKind,
    CapabilityMode,
    CapabilityResult,
)
from jacobian.contracts.workspaces import (
    WorkspaceAttemptDraft,
    WorkspaceBranchId,
    WorkspaceCardId,
    WorkspaceFindingDraft,
    WorkspaceFocusDraft,
    WorkspaceId,
    WorkspaceIdempotencyKey,
    WorkspaceMarkDraft,
    WorkspaceOpenRequest,
    WorkspaceOpenResult,
    WorkspaceQueryRequest,
    WorkspaceQueryResult,
    WorkspaceQueryView,
    WorkspaceRevisionId,
    WorkspaceScratchDraft,
    WorkspaceTag,
    WorkspaceWriteRequest,
    WorkspaceWriteResult,
)

_LOGGER = logging.getLogger(__name__)
CapabilityDescriptionView = Literal["SUMMARY", "CONTRACT", "FULL"]
CapabilityInvocationView = Literal["SUMMARY", "STANDARD", "FULL"]


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
            "workspace.* names are direct MCP tools, not capability IDs; call the "
            "workspace tool directly using its published input schema."
            if capability_id.startswith("workspace.")
            else (
                "Call capability.describe with a mathematical query to search "
                "installed capabilities."
            )
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
    view: CapabilityInvocationView = "STANDARD",
    ctx: Context[AppState, Any] | None = None,
) -> Annotated[CallToolResult, CapabilityResult]:
    active_runtime = _runtime(ctx)
    result = await _invoke_capability_attempt(
        active_runtime,
        capability_id=capability_id,
        payload=payload,
        mode=mode,
        ctx=ctx,
    )
    return _capability_call_tool_result(
        result,
        view=view,
        projection_strategy=_projection_strategy(ctx),
    )


async def workspace_open(
    idempotency_key: WorkspaceIdempotencyKey,
    name: Annotated[str, Field(min_length=1, max_length=128)],
    problem: Annotated[str, Field(min_length=1, max_length=16_384)],
    tags: Annotated[
        list[WorkspaceTag] | None,
        Field(max_length=16),
    ] = None,
    ctx: Context[AppState, Any] | None = None,
) -> WorkspaceOpenResult:
    active_runtime = _runtime(ctx)
    return await _run_blocking(
        active_runtime.core.workspaces.open,
        WorkspaceOpenRequest(
            idempotency_key=idempotency_key,
            name=name,
            problem=problem,
            tags=tuple(tags or ()),
        ),
    )


async def workspace_write(
    workspace_id: Annotated[
        WorkspaceId,
        Field(description="workspace:// handle returned by workspace.open"),
    ],
    branch_id: Annotated[
        WorkspaceBranchId,
        Field(description="branch:// handle returned by workspace.open"),
    ],
    base_revision: Annotated[
        WorkspaceRevisionId,
        Field(
            description=(
                "Exact current revision:// head returned by workspace.open, "
                "workspace.write, or workspace.query."
            )
        ),
    ],
    idempotency_key: Annotated[
        WorkspaceIdempotencyKey,
        Field(
            description=(
                "Caller-chosen key unique to this exact write payload; reuse only "
                "to retry the identical request."
            )
        ),
    ],
    scratch: Annotated[
        list[WorkspaceScratchDraft] | None,
        Field(
            max_length=64,
            description="Optional unverified scratch entries to append.",
        ),
    ] = None,
    findings: Annotated[
        list[WorkspaceFindingDraft] | None,
        Field(
            max_length=64,
            description="Optional typed, unverified cards to append.",
            examples=[
                [
                    {
                        "client_ref": "C1",
                        "kind": "CLAIM",
                        "title": "Candidate conclusion",
                        "body": "Agent-authored reasoning; still unverified.",
                    }
                ]
            ],
        ),
    ] = None,
    attempts: Annotated[
        list[WorkspaceAttemptDraft] | None,
        Field(
            max_length=64,
            description="Optional unverified operational attempts to append.",
            examples=[
                [
                    {
                        "client_ref": "T1",
                        "target_ref": "C1",
                        "method": "direct",
                        "outcome": "COMPLETED",
                        "summary": "The operational attempt finished.",
                    }
                ]
            ],
        ),
    ] = None,
    marks: Annotated[
        list[WorkspaceMarkDraft] | None,
        Field(
            max_length=64,
            description=(
                "Optional append-only lifecycle marks. CLOSED is workflow state, "
                "not proof; RETRACTED and SUPERSEDED deterministically make explicit "
                "dependents stale."
            ),
            examples=[
                [
                    {
                        "client_ref": "M1",
                        "target_ref": "C1",
                        "state": "RETRACTED",
                        "reason": "The recorded premise was withdrawn.",
                    }
                ]
            ],
        ),
    ] = None,
    focus: Annotated[
        WorkspaceFocusDraft | None,
        Field(
            description=(
                "Optional explicit focus update: set active_ref/pinned_refs, use "
                "clear=true to clear, or omit to preserve current focus."
            ),
            examples=[{"active_ref": "C1", "pinned_refs": ["C1"]}],
        ),
    ] = None,
    ctx: Context[AppState, Any] | None = None,
) -> WorkspaceWriteResult:
    active_runtime = _runtime(ctx)
    return await _run_blocking(
        active_runtime.core.workspaces.write,
        WorkspaceWriteRequest(
            idempotency_key=idempotency_key,
            workspace_id=workspace_id,
            branch_id=branch_id,
            base_revision=base_revision,
            scratch=tuple(scratch or ()),
            findings=tuple(findings or ()),
            attempts=tuple(attempts or ()),
            marks=tuple(marks or ()),
            focus=focus,
        ),
    )


async def workspace_query(
    workspace_id: WorkspaceId,
    branch_id: WorkspaceBranchId,
    revision_id: Annotated[
        WorkspaceRevisionId | None,
        Field(
            description=(
                "Optional expected branch head revision:// handle. The query "
                "fails if the current head differs; omit to read the latest."
            )
        ),
    ] = None,
    view: WorkspaceQueryView = WorkspaceQueryView.RESUME,
    target_card_id: WorkspaceCardId | None = None,
    limit: Annotated[StrictInt, Field(ge=1, le=50)] = 10,
    ctx: Context[AppState, Any] | None = None,
) -> WorkspaceQueryResult:
    active_runtime = _runtime(ctx)
    return await _run_blocking(
        active_runtime.core.workspaces.query,
        WorkspaceQueryRequest(
            workspace_id=workspace_id,
            branch_id=branch_id,
            revision_id=revision_id,
            view=view,
            target_card_id=target_card_id,
            limit=limit,
        ),
    )
