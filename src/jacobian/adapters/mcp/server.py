"""Thin MCP 2.0.0b2 adapter over the tested Python kernel."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

from mcp.server.mcpserver.exceptions import ToolError
from mcp.shared.exceptions import MCPError
from mcp_types import ToolAnnotations
from pydantic import AnyHttpUrl, Field, StrictInt

from jacobian import __version__
from jacobian.adapters.mcp.guidance import (
    CAPABILITY_DESCRIBE_DESCRIPTION,
    CAPABILITY_INVOKE_DESCRIPTION,
    OPERATING_GUIDE,
    SERVER_DESCRIPTION,
    SERVER_INSTRUCTIONS,
    discovery_prompt,
    evidence_check_prompt,
)
from jacobian.bounded_process import bounded_process_cancellation
from jacobian.canonical import canonicalize_json
from jacobian.capabilities import CapabilityDiscoveryCursorError, CapabilityPolicy
from jacobian.contracts.capabilities import (
    CapabilityDescriptor,
    CapabilityDiscoveryRequest,
    CapabilityMode,
    CapabilityRequest,
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
CAPABILITY_DISCOVERY_RESPONSE_BYTE_LIMIT = 16_384

_RELATED_CAPABILITIES: dict[str, tuple[tuple[str, str], ...]] = {
    "sat.cnf.materialize": (
        ("sat.model.find", "find a candidate named assignment"),
        ("sat.model.verify", "independently verify a candidate assignment"),
        ("sat.unsat_proof.find", "produce an addition-only DRAT candidate"),
        ("sat.unsat_proof.verify", "independently verify the exact DRAT proof"),
    ),
    "sat.model.find": (
        ("sat.cnf.materialize", "materialize the exact input CNF"),
        ("sat.model.verify", "independently verify the named assignment"),
    ),
    "sat.unsat_proof.find": (
        ("sat.cnf.materialize", "materialize the exact input CNF"),
        ("sat.unsat_proof.verify", "independently verify the retained DRAT proof"),
    ),
    "smt.unsat_proof.find": (
        ("smt.unsat_proof.verify", "independently verify compatible proof evidence"),
        (
            "sat.cnf.materialize",
            "prefer named Boolean CNF for finite colorings and forbidden patterns",
        ),
    ),
    "graph.invariant.maximum_matching.compute": (
        (
            "graph.invariant.maximum_matching.verify",
            "independently replay the stored Tutte-Berge certificate",
        ),
    ),
    "graph.invariant.maximum_matching.verify": (
        (
            "graph.invariant.maximum_matching.compute",
            "produce a matching witness and Tutte-Berge certificate",
        ),
    ),
}

if TYPE_CHECKING:
    from mcp.server import MCPServer
    from mcp.server.mcpserver import Context

    from jacobian.adapters.mcp.remote import TenantKernelRouter
    from jacobian.kernel import JacobianKernel


def _invoke_capability_with_cancellation(
    kernel: Any,
    request: CapabilityRequest,
    cancellation_event: threading.Event,
) -> CapabilityResult:
    with bounded_process_cancellation(cancellation_event):
        result: CapabilityResult = kernel.capabilities.invoke(request)
        return result


def _consume_cancelled_worker_result(task: asyncio.Task[CapabilityResult]) -> None:
    with suppress(BaseException):
        task.result()


def _capability_inspection_extensions(
    capability_id: str,
    descriptors: dict[str, CapabilityDescriptor],
) -> dict[str, Any]:
    extensions: dict[str, Any] = {}
    related = [
        {
            "capability_id": related_id,
            "relationship": relationship,
        }
        for related_id, relationship in _RELATED_CAPABILITIES.get(capability_id, ())
        if related_id in descriptors
    ]
    if related:
        extensions["related_capabilities"] = related
    if capability_id.startswith(("sat.", "smt.")):
        extensions["synchronous_execution"] = {
            "remote_safe_wall_seconds_max": 150,
            "timeout_is_a_non_conclusion": True,
            "partition_larger_searches": True,
            "backend_suitability": (
                "Named Boolean CNF is preferred for finite colorings and forbidden "
                "finite configurations; use SMT when arithmetic or "
                "uninterpreted-function structure is essential."
            ),
        }
    return extensions


def _compact_json_schema(value: Any) -> Any:
    """Drop annotation-only prose while preserving validation semantics."""

    if isinstance(value, dict):
        return {
            key: _compact_json_schema(item)
            for key, item in value.items()
            if key
            not in {
                "$comment",
                "default",
                "deprecated",
                "description",
                "discriminator",
                "examples",
                "readOnly",
                "title",
                "writeOnly",
            }
        }
    if isinstance(value, list):
        return [_compact_json_schema(item) for item in value]
    return value


def _output_schema_summary(schema: dict[str, Any]) -> dict[str, Any]:
    properties = schema.get("properties")
    summary: dict[str, Any] = {
        "type": schema.get("type"),
        "required": schema.get("required", []),
        "property_names": (sorted(properties) if isinstance(properties, dict) else []),
    }
    if "$ref" in schema:
        summary["$ref"] = schema["$ref"]
    if "oneOf" in schema:
        summary["one_of_variants"] = len(schema["oneOf"])
    if "anyOf" in schema:
        summary["any_of_variants"] = len(schema["anyOf"])
    return summary


def _capability_descriptor_view(
    descriptor: CapabilityDescriptor,
    *,
    view: Literal["COMPACT", "FULL"],
) -> dict[str, Any]:
    if view == "FULL":
        return descriptor.model_dump(mode="json")
    runtime = descriptor.provider_runtime
    runtime_summary = (
        runtime.model_dump(
            mode="json",
            exclude_none=True,
            include={
                "availability",
                "version",
                "digest",
                "checker_ids",
                "diagnostic",
            },
        )
        if runtime is not None
        else None
    )
    return {
        "capability_id": descriptor.capability_id,
        "version": descriptor.version,
        "title": descriptor.title,
        "description": descriptor.description,
        "provider": descriptor.provider,
        "provider_runtime": runtime_summary,
        "modes": [mode.value for mode in descriptor.modes],
        "input_schema": _compact_json_schema(descriptor.input_schema),
        "output_schema_summary": _output_schema_summary(descriptor.output_schema),
    }


WORKSPACE_TOOL_NAMES = frozenset(
    {"workspace.open", "workspace.write", "workspace.query"}
)


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


async def _invoke_capability_attempt(
    kernel: Any,
    *,
    capability_id: str,
    payload: dict[str, Any],
    mode: CapabilityMode,
    ctx: Any | None,
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
    worker = asyncio.create_task(
        asyncio.to_thread(
            _invoke_capability_with_cancellation,
            kernel,
            CapabilityRequest(
                capability_id=capability_id,
                mode=mode,
                input=payload,
            ),
            cancellation_event,
        )
    )
    try:
        result = await asyncio.shield(worker)
    except asyncio.CancelledError:
        cancellation_event.set()
        worker.add_done_callback(_consume_cancelled_worker_result)
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


def _catalog_digest(
    catalog_version: str,
    capabilities: tuple[CapabilityDescriptor, ...],
) -> str:
    payload = {
        "catalog_version": catalog_version,
        "capabilities": [
            descriptor.model_dump(mode="json") for descriptor in capabilities
        ],
    }
    return f"sha256:{hashlib.sha256(canonicalize_json(payload)).hexdigest()}"


def _capability_discovery_response(
    kernel: JacobianKernel,
    *,
    query: str | None,
    domain: str | None,
    mode: CapabilityMode | None,
    limit: int | None,
    cursor: str | None,
) -> dict[str, Any]:
    catalog = kernel.capabilities.catalog()
    try:
        discovered = kernel.capabilities.discover(
            CapabilityDiscoveryRequest(
                query=query,
                domain=domain,
                mode=mode,
                limit=limit if limit is not None else 5,
                cursor=cursor,
            )
        )
    except CapabilityDiscoveryCursorError:
        return {
            "error": {
                "code": "INVALID_CURSOR",
                "stage": "capability_discovery",
                "message": "The capability discovery cursor is not in this result set.",
                "hint": (
                    "Restart discovery without a cursor, or reuse the same query, "
                    "domain, mode, and limit that produced next_cursor."
                ),
            }
        }
    response = {
        "kind": "discovery",
        "catalog_version": catalog.catalog_version,
        "policy_profile": catalog.policy_profile,
        "policy_digest": catalog.policy_digest,
        "catalog_digest": _catalog_digest(
            catalog.catalog_version,
            catalog.capabilities,
        ),
        **discovered.model_dump(mode="json"),
        "next_step": {
            "tool": "capability.describe",
            "argument": "capability_id",
            "choose_from": "matches[].capability_id",
        },
        "routing_guidance": {
            "inspect_candidates": (
                "Inspect only the strongest one or two domain-relevant matches; "
                "search again only when none fits the required outcome."
            ),
            "verification_handoff": (
                "Invoke the selected producer before searching for a checker; "
                "follow checker, certificate, and verification fields returned by "
                "the producer result instead of guessing a generic verifier."
            ),
        },
        "response_byte_limit": CAPABILITY_DISCOVERY_RESPONSE_BYTE_LIMIT,
        "truncation_reason": None,
    }
    matches = cast(list[dict[str, Any]], response["matches"])
    while (
        len(canonicalize_json(response)) > CAPABILITY_DISCOVERY_RESPONSE_BYTE_LIMIT
        and len(matches) > 1
    ):
        matches.pop()
        response["truncated"] = True
        response["next_cursor"] = matches[-1]["capability_id"]
        response["truncation_reason"] = "BYTE_LIMIT"
    available_domains = cast(list[str], response["available_domains"])
    response["available_domains_total"] = len(available_domains)
    response["available_domains_truncated"] = False
    while (
        len(canonicalize_json(response)) > CAPABILITY_DISCOVERY_RESPONSE_BYTE_LIMIT
        and available_domains
    ):
        available_domains.pop()
        response["available_domains_truncated"] = True
        response["truncation_reason"] = "BYTE_LIMIT"
    response["match_metadata_truncated"] = False
    compact_fields = ("tags", "matched_on", "matched_terms")
    while len(canonicalize_json(response)) > CAPABILITY_DISCOVERY_RESPONSE_BYTE_LIMIT:
        removed = False
        for match in matches:
            for field in compact_fields:
                values = match.get(field)
                if isinstance(values, list) and values:
                    values.pop()
                    removed = True
                    response["match_metadata_truncated"] = True
                    response["truncation_reason"] = "BYTE_LIMIT"
                    break
            if removed:
                break
        if not removed:
            raise RuntimeError(
                "compact capability discovery response exceeds its hard byte limit"
            )
    return response


def _forbid_extra_tool_arguments(server: Any, *tool_names: str) -> None:
    """Close SDK-generated argument models that otherwise ignore unknown fields."""

    # MCP 2.0.0b2 creates flat function argument models with Pydantic's default
    # ``extra="ignore"``. Workspace writes must reject the entire request instead of
    # silently committing a partial batch when a caller misspells a top-level field.
    manager = server._tool_manager
    for tool_name in tool_names:
        tool = manager.get_tool(tool_name)
        if tool is None:  # pragma: no cover - registration invariant
            raise RuntimeError(f"MCP tool was not registered: {tool_name}")
        argument_model = tool.fn_metadata.arg_model
        argument_model.model_config["extra"] = "forbid"
        argument_model.model_rebuild(force=True)
        tool.parameters = argument_model.model_json_schema(by_alias=True)


def _publish_workspace_normalization_aliases(server: Any) -> None:
    """Advertise exactly the input aliases normalized by workspace contracts."""

    tool = server._tool_manager.get_tool("workspace.write")
    if tool is None:  # pragma: no cover - registration invariant
        raise RuntimeError("workspace tool was not registered: workspace.write")
    schema = tool.parameters
    definitions = schema["$defs"]
    definitions["WorkspaceFindingKind"]["enum"].remove("PROBLEM")
    definitions["WorkspaceFindingKind"]["enum"].append("OPEN_GOAL")
    definitions["WorkspaceAttemptOutcome"]["enum"].append("SUCCEEDED")

    mark_schema = definitions["WorkspaceMarkDraft"]
    reason_schema = mark_schema["properties"]["reason"]
    mark_schema["properties"]["summary"] = {
        **reason_schema,
        "title": "Summary",
        "description": (
            "Input alias for reason. Supplying both summary and reason is rejected."
        ),
    }
    mark_schema["required"].remove("reason")
    mark_schema["oneOf"] = [
        {
            "required": ["reason"],
            "not": {"required": ["summary"]},
        },
        {
            "required": ["summary"],
            "not": {"required": ["reason"]},
        },
    ]


@dataclass(frozen=True, slots=True)
class AppState:
    kernel: JacobianKernel | None
    tenant_router: TenantKernelRouter | None = None


def create_server(
    state_dir: str | Path | None = None,
    *,
    install_references: bool = True,
    tenant_isolation: bool = False,
    allow_anonymous: bool = False,
    anonymous_tenant_id: str = "anonymous",
    token_verifier: Any | None = None,
    auth: Any | None = None,
    capability_adapter_entrypoints: tuple[str, ...] = (),
    capability_exclusions: frozenset[str] = frozenset(),
    capability_policy: CapabilityPolicy | None = None,
    max_tenant_kernels: int | None = None,
) -> MCPServer[AppState]:
    """Create a local or tenant-routed adapter over the Jacobian kernel."""

    if tenant_isolation and capability_exclusions:
        raise ValueError("capability exclusions are supported only by local evaluation")

    # Keep ``--help`` and ``--version`` independent of the MCP runtime's
    # heavier imports and shutdown hooks.
    from mcp.server import MCPServer
    from mcp.server.mcpserver import Context

    from jacobian.adapters.mcp.remote import (
        DEFAULT_MAX_TENANT_KERNELS,
        TenantKernelRouter,
    )
    from jacobian.kernel import JacobianKernel
    from jacobian.references import reference_catalog

    globals().update(
        {
            "Context": Context,
            "JacobianKernel": JacobianKernel,
        }
    )

    class JacobianMCPServer(MCPServer[AppState]):
        async def call_tool(
            self,
            name: str,
            arguments: dict[str, Any],
            context: Context[AppState, Any] | None = None,
        ) -> Any:
            started = time.monotonic()
            argument_digest = _argument_digest(arguments)
            try:
                result = await super().call_tool(name, arguments, context)
            except MCPError:
                _LOGGER.info(
                    "MCP tool call tool=%s status=error duration_ms=%.3f "
                    "response_bytes=0 argument_digest=%s",
                    name,
                    (time.monotonic() - started) * 1000,
                    argument_digest,
                )
                raise
            except Exception as exc:
                _LOGGER.warning(
                    "MCP tool %s failed",
                    name,
                    exc_info=exc,
                )
                _LOGGER.info(
                    "MCP tool call tool=%s status=error duration_ms=%.3f "
                    "response_bytes=0 argument_digest=%s",
                    name,
                    (time.monotonic() - started) * 1000,
                    argument_digest,
                )
                raise ValueError(_public_tool_error(name, exc)) from None
            _LOGGER.info(
                "MCP tool call tool=%s status=success duration_ms=%.3f "
                "response_bytes=%d argument_digest=%s",
                name,
                (time.monotonic() - started) * 1000,
                _response_size(result),
                argument_digest,
            )
            return result

    configured_root = _configured_root(state_dir)
    kernel = (
        None
        if tenant_isolation
        else JacobianKernel(
            configured_root,
            install_references=install_references,
            capability_adapter_entrypoints=capability_adapter_entrypoints,
            capability_exclusions=capability_exclusions,
            capability_policy=capability_policy,
        )
    )
    tenant_router = (
        TenantKernelRouter(
            configured_root,
            install_references=install_references,
            allow_anonymous=allow_anonymous,
            anonymous_tenant_id=anonymous_tenant_id,
            capability_adapter_entrypoints=capability_adapter_entrypoints,
            capability_policy=capability_policy,
            max_tenant_kernels=(
                DEFAULT_MAX_TENANT_KERNELS
                if max_tenant_kernels is None
                else max_tenant_kernels
            ),
        )
        if tenant_isolation
        else None
    )

    @asynccontextmanager
    async def lifespan(_server: MCPServer[AppState]) -> AsyncIterator[AppState]:
        if kernel is not None:
            _start_lean_warmup(kernel)
        yield AppState(kernel=kernel, tenant_router=tenant_router)

    server: MCPServer[AppState] = JacobianMCPServer(
        name="jacobian",
        title="Jacobian Mathematical Workbench",
        description=SERVER_DESCRIPTION,
        instructions=SERVER_INSTRUCTIONS,
        version=__version__,
        lifespan=lifespan,
        token_verifier=token_verifier,
        auth=auth,
    )

    @server.tool(
        name="capability.describe",
        title="Discover mathematical capabilities",
        description=CAPABILITY_DESCRIBE_DESCRIPTION,
        annotations=_tool_annotations(read_only=True, idempotent=True),
        structured_output=True,
    )
    async def capability_describe(
        capability_id: Annotated[
            str | None,
            Field(
                description=(
                    "Exact installed capability ID. When supplied, omit query, "
                    "domain, mode, limit, and cursor."
                )
            ),
        ] = None,
        query: Annotated[
            str | None,
            Field(
                min_length=1,
                max_length=512,
                description=(
                    "Natural-language mathematical outcome to find; no capability "
                    "ID is required."
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
                    "domain, mode, and limit when continuing discovery."
                ),
            ),
        ] = None,
        view: Annotated[
            Literal["COMPACT", "FULL"],
            Field(
                description=(
                    "Exact-lookup projection. COMPACT is the agent-facing default "
                    "with the complete input schema and a concise output/runtime "
                    "summary. FULL returns the complete installed descriptor for "
                    "audit or client generation. Omit for discovery."
                )
            ),
        ] = "COMPACT",
        ctx: Context[AppState, Any] | None = None,
    ) -> dict[str, Any]:
        active_kernel = _kernel(ctx)
        search_arguments = (query, domain, mode, limit, cursor)
        if capability_id is not None and any(
            argument is not None for argument in search_arguments
        ):
            raise AgentRecoveryError(
                "capability_id is an exact lookup and cannot be combined with query, "
                "domain, mode, limit, or cursor. Use one discovery call followed by "
                "one exact description call."
            )
        if capability_id is None:
            return _capability_discovery_response(
                active_kernel,
                query=query,
                domain=domain,
                mode=mode,
                limit=limit,
                cursor=cursor,
            )
        capability_catalog = active_kernel.capabilities.catalog()
        descriptors = {
            item.capability_id: item for item in capability_catalog.capabilities
        }
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
            "invocations": [
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
            ],
        }
        response.update(_capability_inspection_extensions(capability_id, descriptors))
        if capability_id == "lean.check" and active_kernel.lean_checkers:
            response["cache"] = {
                "key": "exact content-addressed certificate and active checker digest",
                "max_entries": 128,
                "warmup_environment_variable": "JACOBIAN_LEAN_WARMUP=1",
                "mathlib_warmup": (
                    active_kernel.lean.mathlib_warmup_health()
                    if active_kernel.lean is not None
                    else {"status": "UNAVAILABLE", "detail": None}
                ),
            }
        return response

    @server.tool(
        name="capability.invoke",
        title="Execute a mathematical capability",
        description=CAPABILITY_INVOKE_DESCRIPTION,
        annotations=_tool_annotations(),
        structured_output=True,
    )
    async def capability_invoke(
        capability_id: str,
        payload: dict[str, Any],
        mode: CapabilityMode = CapabilityMode.EXPLORE,
        ctx: Context[AppState, Any] | None = None,
    ) -> CapabilityResult:
        active_kernel = _kernel(ctx)
        return await _invoke_capability_attempt(
            active_kernel,
            capability_id=capability_id,
            payload=payload,
            mode=mode,
            ctx=ctx,
        )

    @server.tool(
        name="workspace.open",
        description=(
            "Direct tool; do not call capability.describe. Create a durable epistemic "
            "workspace with one canonical problem, a main branch, and an immutable "
            "initial revision. Workspace content is agent-authored and UNVERIFIED."
        ),
        annotations=_tool_annotations(idempotent=True),
        structured_output=False,
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
        active_kernel = _kernel(ctx)
        return await asyncio.to_thread(
            active_kernel.workspaces.open,
            WorkspaceOpenRequest(
                idempotency_key=idempotency_key,
                name=name,
                problem=problem,
                tags=tuple(tags or ()),
            ),
        )

    @server.tool(
        name="workspace.write",
        description=(
            "Direct tool. Do not call capability.describe. Arguments are flat: send "
            "base_revision (never revision_id) and top-level findings, attempts, marks, "
            "scratch, or focus (never a batch wrapper). Every draft uses client_ref, "
            "never ref. Append at an exact base revision. Finding fields are client_ref, "
            "kind, title, body; optional links are dependency_refs and assumption_refs "
            "(never depends_on_refs). Attempt fields are client_ref, target_ref, method, "
            "outcome, summary. Margin marks append an explicit ACTIVE, CLOSED, "
            "RETRACTED, SUPERSEDED, or ARCHIVED state; only SUPERSEDED carries "
            "superseded_by_ref, and summary is accepted as an alias for reason. "
            "References may use a client_ref from the same batch. OPEN_GOAL normalizes "
            "to GOAL and SUCCEEDED normalizes to COMPLETED. PROBLEM is reserved for "
            "workspace.open. A RETRACTED or SUPERSEDED card must receive an ACTIVE mark "
            "before CLOSED or ARCHIVED. Set focus with active_ref/pinned_refs, clear it "
            "with clear=true, or omit it; focus references finding cards, never "
            "attempts, marks, or scratch. Never send verification/assertion/stale "
            "fields: all workspace assertions remain AGENT_RECORDED and UNVERIFIED. "
            "Canonical batch example: "
            'findings=[{"client_ref":"C1","kind":"CLAIM","title":"...","body":"..."}], '
            'attempts=[{"client_ref":"T1","target_ref":"C1","method":"...",'
            '"outcome":"COMPLETED","summary":"..."}], '
            'focus={"active_ref":"C1","pinned_refs":["C1"]}.'
        ),
        annotations=_tool_annotations(idempotent=True),
        structured_output=False,
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
        active_kernel = _kernel(ctx)
        return await asyncio.to_thread(
            active_kernel.workspaces.write,
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

    @server.tool(
        name="workspace.query",
        description=(
            "Direct tool; do not call capability.describe. Read a compact deterministic "
            "RESUME, FRONTIER, ATTEMPTS, CONTEXT, or STALE view from agent-authored "
            "workspace state. CONTEXT requires target_card_id and follows only explicit "
            "dependency/assumption links. Derived staleness is a paper-like warning, "
            "not a mathematical conclusion. Retrieval preserves UNVERIFIED status."
        ),
        annotations=_tool_annotations(read_only=True, idempotent=True),
        structured_output=False,
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
        active_kernel = _kernel(ctx)
        return await asyncio.to_thread(
            active_kernel.workspaces.query,
            WorkspaceQueryRequest(
                workspace_id=workspace_id,
                branch_id=branch_id,
                revision_id=revision_id,
                view=view,
                target_card_id=target_card_id,
                limit=limit,
            ),
        )

    _forbid_extra_tool_arguments(
        server,
        "capability.describe",
        "capability.invoke",
        *WORKSPACE_TOOL_NAMES,
    )
    _publish_workspace_normalization_aliases(server)

    @server.resource(
        "jacobian://instructions",
        name="jacobian-instructions",
        title="Jacobian operating guide",
        description=(
            "Complete guidance for discovering, invoking, and independently checking "
            "Jacobian mathematical capabilities."
        ),
        mime_type="text/markdown",
    )
    async def jacobian_instructions_resource() -> str:
        return OPERATING_GUIDE

    @server.resource(
        "artifact://sha256/{digest}",
        name="artifact",
        description="Read an immutable artifact manifest and payload.",
        mime_type="application/json",
    )
    async def artifact_resource(
        digest: str,
    ) -> str:
        active_kernel = _resource_kernel(kernel, tenant_router)
        artifact = await asyncio.to_thread(
            active_kernel.store.get,
            f"artifact://sha256/{digest}",
        )
        return json.dumps(
            {
                "artifact_uri": artifact.artifact_uri,
                "manifest": artifact.manifest.model_dump(mode="json"),
                "payload": artifact.payload,
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    @server.resource(
        "capability://catalog",
        name="capability-catalog",
        description=(
            "Installed model-facing operations, supported lanes, and compact schemas."
        ),
        mime_type="application/json",
    )
    async def capability_catalog_resource() -> str:
        active_kernel = _resource_kernel(kernel, tenant_router)
        return json.dumps(
            active_kernel.capabilities.catalog().model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
        )

    @server.resource(
        "reference://catalog",
        name="reference-catalog",
        description="Read installed domain schema, semantics, plugin, and checker IDs.",
        mime_type="application/json",
    )
    async def reference_catalog_resource() -> str:
        active_kernel = _resource_kernel(kernel, tenant_router)
        return json.dumps(
            reference_catalog(
                active_kernel.references,
                graph=active_kernel.graph,
                polytope=active_kernel.polytope,
                polytope_checkers=active_kernel.polytope_checkers,
                polynomial=active_kernel.polynomial,
                universal_algebra=active_kernel.universal_algebra,
                lean=active_kernel.lean_checkers,
            ),
            ensure_ascii=False,
            sort_keys=True,
        )

    @server.resource(
        "experiment://{experiment_id}",
        name="experiment",
        description="Read the latest durable experiment snapshot.",
        mime_type="application/json",
    )
    async def experiment_resource(
        experiment_id: str,
    ) -> str:
        active_kernel = _resource_kernel(kernel, tenant_router)
        snapshot = await asyncio.to_thread(
            active_kernel.experiment_router.inspect,
            f"experiment://{experiment_id}",
        )
        return json.dumps(
            snapshot.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
        )

    @server.resource(
        "experiment://{experiment_id}/accounting",
        name="experiment-accounting",
        description="Read durable enumeration accounting and assurance labels.",
        mime_type="application/json",
    )
    async def experiment_accounting_resource(
        experiment_id: str,
    ) -> str:
        active_kernel = _resource_kernel(kernel, tenant_router)
        snapshot = await asyncio.to_thread(
            active_kernel.experiment_router.inspect,
            f"experiment://{experiment_id}",
        )
        coverage = getattr(snapshot, "coverage", None)
        return json.dumps(
            {
                "experiment_uri": snapshot.experiment_uri,
                "state": snapshot.state.value,
                "stop_reason": (
                    snapshot.stop_reason.value
                    if snapshot.stop_reason is not None
                    else None
                ),
                "coverage": coverage.value if coverage is not None else None,
                "verification": snapshot.verification.value,
                "accounting": snapshot.accounting.model_dump(mode="json"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    @server.resource(
        "experiment://{experiment_id}/scope",
        name="experiment-scope",
        description="Read the current enumeration scope artifact, when available.",
        mime_type="application/json",
    )
    async def experiment_scope_resource(
        experiment_id: str,
    ) -> str:
        active_kernel = _resource_kernel(kernel, tenant_router)
        snapshot = await asyncio.to_thread(
            active_kernel.experiment_router.inspect,
            f"experiment://{experiment_id}",
        )
        return await asyncio.to_thread(
            _experiment_scope_content,
            active_kernel,
            snapshot,
        )

    @server.resource(
        "experiment://{experiment_id}/archive",
        name="experiment-archive",
        description="Read the immutable archive manifest and page handles.",
        mime_type="application/json",
    )
    async def experiment_archive_resource(
        experiment_id: str,
    ) -> str:
        active_kernel = _resource_kernel(kernel, tenant_router)
        snapshot = await asyncio.to_thread(
            active_kernel.experiment_router.inspect,
            f"experiment://{experiment_id}",
        )
        if snapshot.archive_uri is None:
            return json.dumps(
                {
                    "experiment_uri": snapshot.experiment_uri,
                    "archive_uri": None,
                    "page_uris": list(snapshot.archive_page_uris),
                },
                sort_keys=True,
            )
        archive = await asyncio.to_thread(
            active_kernel.store.get,
            snapshot.archive_uri,
        )
        return json.dumps(
            {
                "experiment_uri": snapshot.experiment_uri,
                "archive_uri": archive.artifact_uri,
                "manifest": archive.manifest.model_dump(mode="json"),
                "payload": archive.payload,
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    @server.prompt(
        name="jacobian-discover",
        title="Discover Jacobian capabilities",
        description=(
            "Guide capability discovery without choosing the agent's mathematical "
            "research strategy."
        ),
    )
    def jacobian_discover_prompt(
        task: Annotated[
            str,
            Field(description="The mathematical task or desired outcome."),
        ],
    ) -> str:
        return discovery_prompt(task)

    @server.prompt(
        name="jacobian-check-evidence",
        title="Check mathematical evidence with Jacobian",
        description=(
            "Guide selection and use of an installed independent checker without "
            "promoting unverified evidence."
        ),
    )
    def jacobian_check_evidence_prompt(
        claim: Annotated[
            str,
            Field(description="The exact mathematical claim to check."),
        ],
        artifact_uri: Annotated[
            str | None,
            Field(description="Optional artifact:// URI carrying candidate evidence."),
        ] = None,
    ) -> str:
        return evidence_check_prompt(claim, artifact_uri)

    return server


def _kernel(ctx: Context[AppState, Any] | None) -> JacobianKernel:
    if ctx is None:
        raise AgentRecoveryError(
            "Jacobian is unavailable for this request. Retry once; if it fails "
            "again, inspect the local Jacobian log."
        )
    state = ctx.request_context.lifespan_context
    if not isinstance(state, AppState):
        raise AgentRecoveryError(
            "Jacobian is unavailable for this request. Retry once; if it fails "
            "again, inspect the local Jacobian log."
        )
    if state.tenant_router is not None:
        from mcp.server.auth.middleware.auth_context import get_access_token

        access_token = get_access_token()
        subject = access_token.subject if access_token is not None else None
        kernel = state.tenant_router.kernel_for(subject)
        _start_lean_warmup(kernel)
        return kernel
    if state.kernel is None:
        raise AgentRecoveryError(
            "Jacobian is unavailable for this request. Retry once; if it fails "
            "again, inspect the local Jacobian log."
        )
    return state.kernel


def _start_lean_warmup(kernel: JacobianKernel) -> None:
    if kernel.lean is not None and os.environ.get("JACOBIAN_LEAN_WARMUP") == "1":
        kernel.lean.start_mathlib_warmup()


def _resource_kernel(
    kernel: JacobianKernel | None,
    tenant_router: TenantKernelRouter | None,
) -> JacobianKernel:
    """Route resources through the same auth context as tools.

    MCP 2.0.0b2 does not inject ``Context`` into static resources, but its HTTP
    authentication middleware still scopes the access token with a contextvar.
    """

    if tenant_router is not None:
        from mcp.server.auth.middleware.auth_context import get_access_token

        access_token = get_access_token()
        subject = access_token.subject if access_token is not None else None
        return tenant_router.kernel_for(subject)
    if kernel is None:
        raise AgentRecoveryError(
            "Jacobian is unavailable for this resource request. Retry once; if it "
            "fails again, inspect the local Jacobian log."
        )
    return kernel


def _configured_root(state_dir: str | Path | None) -> Path:
    if state_dir is not None:
        return Path(state_dir)
    return Path(os.environ.get("JACOBIAN_STATE_DIR", ".jacobian"))


def _public_tool_error(tool_name: str, exc: Exception) -> str:
    from jacobian.adapters.mcp.remote import (
        AuthenticationError,
        TenantKernelLimitError,
    )
    from jacobian.experiments import ExperimentNotFoundError
    from jacobian.registry import CheckerNotFoundError
    from jacobian.store import ArtifactNotFoundError
    from jacobian.workspaces import (
        WorkspaceConflictError,
        WorkspaceIdempotencyError,
        WorkspaceNotFoundError,
        WorkspaceReferenceError,
    )

    tool_error = exc.__cause__ if isinstance(exc, ToolError) else exc
    if not isinstance(tool_error, Exception):
        tool_error = exc
    if isinstance(tool_error, AgentRecoveryError):
        code = "SERVICE_UNAVAILABLE"
        message = str(tool_error)
        hint = "Follow the recovery action in the message, then retry the tool."
    elif isinstance(tool_error, TimeoutError):
        code = "OPERATION_TIMED_OUT"
        message = "The operation did not finish within the allowed time."
        hint = "Retry with a larger time budget or a smaller request."
    elif isinstance(tool_error, AuthenticationError):
        code = "AUTHENTICATION_REQUIRED"
        message = str(tool_error)
        hint = "Authenticate with a configured bearer token, then retry."
    elif isinstance(tool_error, TenantKernelLimitError):
        code = "TENANT_KERNEL_LIMIT"
        message = str(tool_error)
        hint = (
            "Retry on another server instance or ask the operator to raise the limit."
        )
    elif isinstance(tool_error, PermissionError):
        code = "PERMISSION_DENIED"
        message = "Jacobian could not access the required local resource."
        hint = "Check the state-directory permissions, then retry."
    elif isinstance(
        tool_error,
        (
            ArtifactNotFoundError,
            CheckerNotFoundError,
            ExperimentNotFoundError,
            WorkspaceNotFoundError,
        ),
    ):
        code = "RESOURCE_NOT_FOUND"
        message = "A required Jacobian resource was not found."
        hint = (
            "Check the artifact or experiment URI returned by the earlier tool call, "
            "then retry."
        )
    elif isinstance(tool_error, WorkspaceConflictError):
        code = "WORKSPACE_CONFLICT"
        message = str(tool_error)
        hint = "Query the latest workspace revision, then retry from that exact head."
    elif isinstance(
        tool_error,
        (WorkspaceIdempotencyError, WorkspaceReferenceError),
    ):
        code = "INVALID_INPUT"
        message = str(tool_error)
        hint = "Check the published workspace tool schema and returned handles."
    elif isinstance(tool_error, ValueError):
        code = "INVALID_INPUT"
        message = "The tool input is not valid for this operation."
        hint = (
            "Check the published workspace tool schema, then retry."
            if tool_name.startswith("workspace.")
            else "Check the tool input schema or call capability.describe, then retry."
        )
    else:
        code = "OPERATION_FAILED"
        message = "Jacobian could not complete the operation."
        hint = "Retry once; if it fails again, inspect the local Jacobian log."
    return json.dumps(
        {
            "error": {
                "code": code,
                "stage": tool_name,
                "message": message,
                "hint": hint,
            }
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _experiment_scope_content(kernel: JacobianKernel, snapshot: Any) -> str:
    scope_uri = getattr(snapshot, "scope_uri", None)
    if scope_uri is None:
        return json.dumps(
            {
                "experiment_uri": snapshot.experiment_uri,
                "scope_uri": None,
            },
            sort_keys=True,
        )
    scope = kernel.store.get(scope_uri)
    return json.dumps(
        {
            "experiment_uri": snapshot.experiment_uri,
            "scope_uri": scope.artifact_uri,
            "manifest": scope.manifest.model_dump(mode="json"),
            "payload": scope.payload,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="jacobian-mcp",
        description="Run the Jacobian MCP server locally or over remote HTTP.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http", "sse"),
        default="stdio",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        help="state root; defaults to JACOBIAN_STATE_DIR or .jacobian",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--path", default="/mcp")
    parser.add_argument(
        "--stateless-http",
        action="store_true",
        help="use stateless Streamable HTTP sessions",
    )
    parser.add_argument(
        "--auth-tokens-file",
        type=Path,
        help="JSON secret mapping opaque bearer tokens to tenant IDs",
    )
    parser.add_argument(
        "--public-base-url",
        help="public issuer/resource base URL advertised to remote clients",
    )
    parser.add_argument(
        "--allow-anonymous",
        action="store_true",
        help="development only: permit unauthenticated remote requests",
    )
    parser.add_argument(
        "--anonymous-tenant-id",
        default="anonymous",
        help=(
            "fixed operator-chosen tenant namespace for anonymous mode; use a "
            "different value for each isolated test endpoint"
        ),
    )
    parser.add_argument(
        "--capability-adapter",
        action="append",
        default=[],
        help="operator-approved package.module:factory entrypoint; repeatable",
    )
    parser.add_argument(
        "--capability-policy-profile",
        choices=("DEFAULT", "COMPUTE_VERIFY_NO_RETRIEVAL"),
        default="DEFAULT",
        help=(
            "operator capability policy profile; the no-retrieval profile is "
            "intended for compute/verify evaluation isolation"
        ),
    )
    for option, destination, help_text in (
        (
            "--allow-capability",
            "allowed_capability_ids",
            "allow only this capability ID; repeatable",
        ),
        (
            "--deny-capability",
            "denied_capability_ids",
            "deny this capability ID; repeatable",
        ),
        ("--allow-domain", "allowed_domains", "allow only this domain; repeatable"),
        ("--deny-domain", "denied_domains", "deny this domain; repeatable"),
        ("--allow-tag", "allowed_tags", "allow only capabilities with this tag"),
        ("--deny-tag", "denied_tags", "deny capabilities with this tag"),
    ):
        parser.add_argument(
            option,
            dest=destination,
            action="append",
            default=[],
            help=help_text,
        )
    parser.add_argument(
        "--allow-mode",
        action="append",
        default=[],
        choices=tuple(mode.value for mode in CapabilityMode),
        help="allow only this capability mode; repeatable",
    )
    parser.add_argument(
        "--deny-mode",
        action="append",
        default=[],
        choices=tuple(mode.value for mode in CapabilityMode),
        help="deny this capability mode; repeatable",
    )
    parser.add_argument(
        "--max-tenant-kernels",
        type=int,
        default=32,
        help="maximum in-memory tenant kernels for remote transports",
    )
    args = parser.parse_args()
    if args.max_tenant_kernels < 1:
        parser.error("--max-tenant-kernels must be positive")
    if args.anonymous_tenant_id != "anonymous" and not args.allow_anonymous:
        parser.error("--anonymous-tenant-id requires --allow-anonymous")
    args.path = args.path if args.path.startswith("/") else f"/{args.path}"
    capability_policy = CapabilityPolicy(
        profile=args.capability_policy_profile,
        allowed_capability_ids=frozenset(args.allowed_capability_ids),
        denied_capability_ids=frozenset(args.denied_capability_ids),
        allowed_domains=frozenset(args.allowed_domains),
        denied_domains=frozenset(args.denied_domains),
        allowed_tags=frozenset(args.allowed_tags),
        denied_tags=frozenset(args.denied_tags),
        allowed_modes=frozenset(CapabilityMode(value) for value in args.allow_mode),
        denied_modes=frozenset(CapabilityMode(value) for value in args.deny_mode),
    )
    if args.transport == "stdio":
        if (
            args.auth_tokens_file is not None
            or args.allow_anonymous
            or args.anonymous_tenant_id != "anonymous"
        ):
            parser.error("remote authentication options cannot be used with stdio")
        create_server(
            state_dir=args.state_dir,
            capability_adapter_entrypoints=tuple(args.capability_adapter),
            capability_policy=capability_policy,
        ).run("stdio")
        return

    if args.auth_tokens_file is None and not args.allow_anonymous:
        parser.error(
            "remote transports require --auth-tokens-file or explicit --allow-anonymous"
        )
    token_verifier = None
    auth = None
    if args.auth_tokens_file is not None:
        from mcp.server.auth.settings import AuthSettings

        from jacobian.adapters.mcp.remote import (
            StaticTokenVerifier,
            load_static_token_file,
        )

        public_base_url = str(
            args.public_base_url or f"http://{args.host}:{args.port}"
        ).rstrip("/")
        token_verifier = StaticTokenVerifier(
            load_static_token_file(args.auth_tokens_file)
        )
        auth = AuthSettings(
            issuer_url=AnyHttpUrl(public_base_url),
            resource_server_url=AnyHttpUrl(f"{public_base_url}{args.path}"),
            required_scopes=["jacobian:use"],
        )
    server = create_server(
        state_dir=args.state_dir,
        tenant_isolation=True,
        allow_anonymous=args.allow_anonymous,
        anonymous_tenant_id=args.anonymous_tenant_id,
        token_verifier=token_verifier,
        auth=auth,
        capability_adapter_entrypoints=tuple(args.capability_adapter),
        capability_policy=capability_policy,
        max_tenant_kernels=args.max_tenant_kernels,
    )
    if args.transport == "streamable-http":
        server.run(
            "streamable-http",
            host=args.host,
            port=args.port,
            streamable_http_path=args.path,
            stateless_http=args.stateless_http,
        )
    else:
        server.run(
            "sse",
            host=args.host,
            port=args.port,
            sse_path=args.path,
            message_path="/messages/",
        )


if __name__ == "__main__":
    main()
