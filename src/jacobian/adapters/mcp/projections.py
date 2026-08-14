"""Bounded search projection and cancellation helpers for the MCP adapter."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, cast

from jacobian.adapters.mcp.constants import (
    OPERATION_DISCOVERY_RESPONSE_BYTE_LIMIT,
)
from jacobian.contracts.operations import (
    OperationDescriptor,
    OperationDiscoveryRequest,
    OperationInputKind,
)
from jacobian.operation_errors import OperationDiscoveryCursorError

_LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


def _mcp_text_json_bytes(value: object) -> bytes:
    """Measure JSON as FastMCP renders structured tool results."""
    return json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")


def _discovery_operation_card(
    match: dict[str, Any],
    descriptor: OperationDescriptor,
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
            runtime.availability.value
            if runtime is not None
            else ("AVAILABLE" if descriptor.provider == "built-in" else "UNKNOWN")
        ),
    }


def _operation_discovery_response(
    runtime: Any,
    *,
    query: str,
    domain: str | None,
    input_kind: OperationInputKind | None,
    artifact_type: str | None,
    limit: int | None,
    cursor: str | None,
) -> dict[str, Any]:
    discovery_request = OperationDiscoveryRequest(
        query=query,
        domain=domain,
        input_kind=input_kind,
        artifact_type=artifact_type,
        limit=limit if limit is not None else 5,
        cursor=cursor,
    )
    try:
        operations = getattr(getattr(runtime, "core", None), "operations", runtime)
        discovered = operations.search(discovery_request)
    except OperationDiscoveryCursorError:
        return {
            "error": {
                "code": "INVALID_CURSOR",
                "stage": "operation_discovery",
                "message": "The operation discovery cursor is not in this result set.",
                "hint": (
                    "Restart discovery without a cursor, or reuse the same query, "
                    "domain, input_kind, artifact_type, and limit that produced "
                    "next_cursor."
                ),
            }
        }
    discovered_payload = discovered.model_dump(mode="json")
    discovered_payload["matches"] = [
        _discovery_operation_card(
            match,
            cast(
                OperationDescriptor,
                operations.inspect(match["operation_id"]),
            ),
        )
        for match in cast(list[dict[str, Any]], discovered_payload["matches"])
    ]
    response = {
        "kind": "discovery",
        **discovered_payload,
        "catalog_resource": "operation://catalog",
        "response_byte_limit": OPERATION_DISCOVERY_RESPONSE_BYTE_LIMIT,
        "truncation_reason": None,
        "match_metadata_truncated": False,
    }
    matches = cast(list[dict[str, Any]], response["matches"])
    while (
        len(_mcp_text_json_bytes(response)) > OPERATION_DISCOVERY_RESPONSE_BYTE_LIMIT
        and len(matches) > 1
    ):
        matches.pop()
        response["truncated"] = True
        response["next_cursor"] = matches[-1]["operation_id"]
        response["truncation_reason"] = "BYTE_LIMIT"
    compact_fields = (
        "tags",
        "produced_artifact_types",
    )
    while len(_mcp_text_json_bytes(response)) > OPERATION_DISCOVERY_RESPONSE_BYTE_LIMIT:
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
                "compact operation discovery response exceeds its hard byte limit"
            )
    return response
