"""JSON projections and capability discovery views for the MCP adapter."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from typing import TYPE_CHECKING, Any, Literal, cast

from jacobian.adapters.mcp.constants import CAPABILITY_DISCOVERY_RESPONSE_BYTE_LIMIT
from jacobian.bounded_process import bounded_process_cancellation
from jacobian.canonical import canonicalize_json
from jacobian.capability_service import CapabilityDiscoveryCursorError
from jacobian.contracts.capabilities import (
    CapabilityDescriptor,
    CapabilityDiscoveryRequest,
    CapabilityInputKind,
    CapabilityMode,
    CapabilityRequest,
    CapabilityResult,
)

_LOGGER = logging.getLogger(__name__)
CapabilityDescriptionView = Literal["SUMMARY", "CONTRACT", "FULL"]
_DISCOVERY_INVOCATION_EXAMPLE_BYTE_LIMIT = 2 * 1024

if TYPE_CHECKING:
    from jacobian.runtime.model import JacobianRuntime


def _mcp_text_json_bytes(value: object) -> bytes:
    """Measure JSON as FastMCP renders structured tool results."""
    return json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")


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
            "materialize named Boolean CNF for finite colorings and forbidden patterns",
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
    "graph.hamiltonian_path.decide": (
        (
            "graph.hamiltonian_path.verify",
            "independently verify the stored positive or negative decision",
        ),
    ),
    "graph.hamiltonian_path.verify": (
        (
            "graph.hamiltonian_path.decide",
            "produce a complete bounded decision and optional path witness",
        ),
    ),
    "polynomial.jacobian_syzygy.minimum_degree.compute": (
        (
            "polynomial.jacobian_syzygy.minimum_degree.verify",
            "independently rebuild the graded maps, ranks, minors, and first kernel",
        ),
    ),
    "polynomial.jacobian_syzygy.minimum_degree.verify": (
        (
            "polynomial.jacobian_syzygy.minimum_degree.compute",
            "produce the provenance-bound graded rank ledger and kernel witness",
        ),
    ),
    "geometry.projective_line_arrangement.flats.materialize": (
        (
            "geometry.projective_line_arrangement.flats.verify",
            "independently rebuild all projective flats and pair accounting",
        ),
    ),
    "geometry.projective_line_arrangement.flats.verify": (
        (
            "geometry.projective_line_arrangement.flats.materialize",
            "materialize normalized lines, exact flats, incidences and multiplicities",
        ),
    ),
}


def _invoke_capability_with_cancellation(
    runtime: Any,
    request: CapabilityRequest,
    cancellation_event: threading.Event,
) -> CapabilityResult:
    with bounded_process_cancellation(cancellation_event):
        result: CapabilityResult = runtime.core.capabilities.invoke(request)
        return result


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
            "larger_search_requires_multiple_bounded_invocations": True,
            "backend_suitability": (
                "Named Boolean CNF represents finite colorings and forbidden finite "
                "configurations; SMT represents arithmetic and uninterpreted-function "
                "structure."
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


def _input_schema_summary(schema: dict[str, Any]) -> dict[str, Any]:
    properties = schema.get("properties")
    return {
        "type": schema.get("type"),
        "required": schema.get("required", []),
        "property_names": sorted(properties) if isinstance(properties, dict) else [],
    }


def _discovery_operation_card(
    match: dict[str, Any],
    descriptor: CapabilityDescriptor,
    descriptors: dict[str, CapabilityDescriptor],
) -> dict[str, Any]:
    """Add compact decision facts without recommending a research action."""

    runtime = descriptor.provider_runtime
    related = [
        {
            "capability_id": related_id,
            "relationship": relationship,
        }
        for related_id, relationship in _RELATED_CAPABILITIES.get(
            descriptor.capability_id, ()
        )
        if related_id in descriptors
    ]
    invocation_example = None
    if descriptor.invocation_examples:
        example = descriptor.invocation_examples[0]
        candidate = {
            "mode": example.mode.value,
            "payload": example.input,
        }
        if (
            len(canonicalize_json(candidate))
            <= _DISCOVERY_INVOCATION_EXAMPLE_BYTE_LIMIT
        ):
            invocation_example = candidate
    return {
        **match,
        "accepted_input_kinds": [
            kind.value for kind in descriptor.accepted_input_kinds
        ],
        "accepted_artifact_types": list(descriptor.accepted_artifact_types),
        "produced_artifact_types": list(descriptor.produced_artifact_types),
        "output_schema_summary": _output_schema_summary(descriptor.output_schema),
        "scope": "EXACT_SUPPLIED_INPUT_OR_CLAIM",
        "assurance_ceiling": (
            "VERIFIED" if CapabilityMode.VERIFY in descriptor.modes else "COMPUTED"
        ),
        "provider_availability": (
            runtime.availability.value if runtime is not None else "UNKNOWN"
        ),
        "related_capabilities": related,
        **(
            {"invocation_example": invocation_example}
            if invocation_example is not None
            else {
                "input_schema_summary": _input_schema_summary(descriptor.input_schema)
            }
        ),
    }


def _discovery_recovery_paths(
    request: CapabilityDiscoveryRequest,
    *,
    domain_filter_status: str,
    portfolio_fit: str,
    routing_status: str,
) -> list[dict[str, Any]]:
    """Return unranked access choices for weak, empty, or incompatible discovery."""

    if (
        domain_filter_status != "UNKNOWN"
        and portfolio_fit not in {"ONLY_WEAK_LEXICAL_MATCHES", "NO_LEXICAL_MATCHES"}
        and routing_status != "NO_ROUTE"
    ):
        return []
    paths: list[dict[str, Any]] = []
    if request.query is not None:
        paths.append(
            {
                "action": "reformulate_query",
                "tool": "math.find",
                "change": "Use different or broader mathematical language for query.",
            }
        )
    if domain_filter_status == "UNKNOWN":
        paths.append(
            {
                "action": "remove_unknown_domain_filter",
                "tool": "math.find",
                "rejected_domain": request.domain,
                "change": "Retry without the unrecognized domain filter.",
            }
        )
    if domain_filter_status != "UNKNOWN" and any(
        value is not None
        for value in (
            request.domain,
            request.mode,
            request.input_kind,
            request.artifact_type,
        )
    ):
        paths.append(
            {
                "action": "remove_filters",
                "tool": "math.find",
                "change": "Remove domain, mode, input_kind, or artifact_type filters.",
            }
        )
    paths.extend(
        (
            {
                "action": "browse",
                "tool": "math.find",
                "arguments": {},
            },
            {
                "action": "inspect_catalog",
                "resource_uri": "capability://catalog",
            },
        )
    )
    return paths


def _capability_descriptor_view(
    descriptor: CapabilityDescriptor,
    *,
    view: CapabilityDescriptionView,
) -> dict[str, Any]:
    if view == "FULL":
        return descriptor.model_dump(mode="json")
    runtime = descriptor.provider_runtime
    if view == "SUMMARY":
        runtime_summary = (
            runtime.model_dump(
                mode="json",
                exclude_none=True,
                include={
                    "availability",
                    "version",
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
            "tags": list(descriptor.tags),
            "accepted_input_kinds": [
                kind.value for kind in descriptor.accepted_input_kinds
            ],
            "accepted_artifact_types": list(descriptor.accepted_artifact_types),
            "produced_artifact_types": list(descriptor.produced_artifact_types),
            "input_schema_summary": _input_schema_summary(descriptor.input_schema),
            "output_schema_summary": _output_schema_summary(descriptor.output_schema),
            "has_invocation_examples": bool(descriptor.invocation_examples),
        }
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
        "accepted_input_kinds": [
            kind.value for kind in descriptor.accepted_input_kinds
        ],
        "accepted_artifact_types": list(descriptor.accepted_artifact_types),
        "produced_artifact_types": list(descriptor.produced_artifact_types),
        "input_schema": _compact_json_schema(descriptor.input_schema),
        "output_schema_summary": _output_schema_summary(descriptor.output_schema),
    }


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
    runtime: JacobianRuntime,
    *,
    query: str | None,
    domain: str | None,
    mode: CapabilityMode | None,
    input_kind: CapabilityInputKind | None,
    artifact_type: str | None,
    limit: int | None,
    cursor: str | None,
) -> dict[str, Any]:
    catalog = runtime.core.capabilities.catalog()
    discovery_request = CapabilityDiscoveryRequest(
        query=query,
        domain=domain,
        mode=mode,
        input_kind=input_kind,
        artifact_type=artifact_type,
        limit=limit if limit is not None else 5,
        cursor=cursor,
    )
    try:
        discovered = runtime.core.capabilities.discover(discovery_request)
    except CapabilityDiscoveryCursorError:
        return {
            "error": {
                "code": "INVALID_CURSOR",
                "stage": "capability_discovery",
                "message": "The capability discovery cursor is not in this result set.",
                "hint": (
                    "Restart discovery without a cursor, or reuse the same query, "
                    "domain, mode, input_kind, artifact_type, and limit that produced "
                    "next_cursor."
                ),
            }
        }
    descriptors = {
        descriptor.capability_id: descriptor for descriptor in catalog.capabilities
    }
    discovered_payload = discovered.model_dump(mode="json")
    discovered_payload["matches"] = [
        _discovery_operation_card(
            match, descriptors[match["capability_id"]], descriptors
        )
        for match in cast(list[dict[str, Any]], discovered_payload["matches"])
    ]
    recovery_paths = _discovery_recovery_paths(
        discovery_request,
        domain_filter_status=discovered.domain_filter_status,
        portfolio_fit=discovered.portfolio_fit,
        routing_status=discovered.routing_status,
    )
    response = {
        "kind": "discovery",
        "catalog_version": catalog.catalog_version,
        "policy_profile": catalog.policy_profile,
        "policy_digest": catalog.policy_digest,
        "catalog_digest": _catalog_digest(
            catalog.catalog_version,
            catalog.capabilities,
        ),
        **discovered_payload,
        "available_recovery_paths": recovery_paths,
        "recovery_paths_are_unranked": True,
        "response_byte_limit": CAPABILITY_DISCOVERY_RESPONSE_BYTE_LIMIT,
        "truncation_reason": None,
    }
    matches = cast(list[dict[str, Any]], response["matches"])
    while (
        len(_mcp_text_json_bytes(response)) > CAPABILITY_DISCOVERY_RESPONSE_BYTE_LIMIT
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
        len(_mcp_text_json_bytes(response)) > CAPABILITY_DISCOVERY_RESPONSE_BYTE_LIMIT
        and available_domains
    ):
        available_domains.pop()
        response["available_domains_truncated"] = True
        response["truncation_reason"] = "BYTE_LIMIT"
    response["match_metadata_truncated"] = False
    compact_fields = (
        "tags",
        "matched_on",
        "matched_terms",
        "produced_artifact_types",
    )
    while (
        len(_mcp_text_json_bytes(response)) > CAPABILITY_DISCOVERY_RESPONSE_BYTE_LIMIT
    ):
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
