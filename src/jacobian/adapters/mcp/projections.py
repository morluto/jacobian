"""Bounded search projection and cancellation helpers for the MCP adapter."""

from __future__ import annotations

import json
import logging
import threading
from typing import TYPE_CHECKING, Any, cast

from jacobian.adapters.mcp.constants import (
    CAPABILITY_DISCOVERY_RESPONSE_BYTE_LIMIT,
)
from jacobian.bounded_process import bounded_process_cancellation
from jacobian.capability_service import CapabilityDiscoveryCursorError
from jacobian.contracts.capabilities import (
    CapabilityDescriptor,
    CapabilityDiscoveryRequest,
    CapabilityInputKind,
    CapabilityRequest,
    CapabilityResult,
)

_LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from jacobian.runtime.model import JacobianRuntime


def _mcp_text_json_bytes(value: object) -> bytes:
    """Measure JSON as FastMCP renders structured tool results."""
    return json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")


def _invoke_capability_with_cancellation(
    runtime: Any,
    request: CapabilityRequest,
    cancellation_event: threading.Event,
) -> CapabilityResult:
    with bounded_process_cancellation(cancellation_event):
        result: CapabilityResult = runtime.core.capabilities.invoke(request)
        return result


def _discovery_operation_card(
    match: dict[str, Any],
    descriptor: CapabilityDescriptor,
) -> dict[str, Any]:
    """Add installed availability and typed routing facts to lexical search."""

    runtime = descriptor.provider_runtime
    return {
        **match,
        "accepted_input_kinds": [
            kind.value for kind in descriptor.accepted_input_kinds
        ],
        "accepted_artifact_types": list(descriptor.accepted_artifact_types),
        "produced_artifact_types": list(descriptor.produced_artifact_types),
        "input_ports": [
            port.model_dump(mode="json") for port in descriptor.input_ports
        ],
        "output_ports": [
            port.model_dump(mode="json") for port in descriptor.output_ports
        ],
        "provider_availability": (
            runtime.availability.value if runtime is not None else "UNKNOWN"
        ),
    }


def _capability_discovery_response(
    runtime: JacobianRuntime,
    *,
    query: str,
    domain: str | None,
    input_kind: CapabilityInputKind | None,
    artifact_type: str | None,
    limit: int | None,
    cursor: str | None,
) -> dict[str, Any]:
    catalog = runtime.core.capabilities.catalog()
    discovery_request = CapabilityDiscoveryRequest(
        query=query,
        domain=domain,
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
                    "domain, input_kind, artifact_type, and limit that produced "
                    "next_cursor."
                ),
            }
        }
    descriptors = {
        descriptor.capability_id: descriptor for descriptor in catalog.capabilities
    }
    discovered_payload = discovered.model_dump(mode="json")
    discovered_payload["matches"] = [
        _discovery_operation_card(match, descriptors[match["capability_id"]])
        for match in cast(list[dict[str, Any]], discovered_payload["matches"])
    ]
    response = {
        "kind": "discovery",
        **discovered_payload,
        "catalog_resource": "capability://catalog",
        "response_byte_limit": CAPABILITY_DISCOVERY_RESPONSE_BYTE_LIMIT,
        "truncation_reason": None,
        "match_metadata_truncated": False,
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
    compact_fields = (
        "tags",
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
