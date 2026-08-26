"""Bounded search projection and cancellation helpers for the MCP adapter."""

from __future__ import annotations

import json
from typing import Any, cast

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDiscoveryRequest
from jacobian.catalog.search import OperationDiscoveryCursorError

_DISCOVERY_RESPONSE_BYTE_LIMIT = 16_384


def _mcp_text_json_bytes(value: object) -> bytes:
    """Measure JSON as the MCP SDK renders structured tool results."""

    return json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")


def _operation_discovery_response(
    catalog: Catalog,
    *,
    query: str,
    domain: str | None,
    limit: int | None,
    cursor: str | None,
) -> dict[str, Any]:
    discovery_request = OperationDiscoveryRequest(
        query=query,
        domain=domain,
        limit=limit if limit is not None else 5,
        cursor=cursor,
    )
    try:
        discovered = catalog.search(discovery_request)
    except OperationDiscoveryCursorError:
        return {
            "kind": "error",
            "error": {
                "code": "INVALID_CURSOR",
                "stage": "operation_discovery",
                "message": "The operation discovery cursor is not in this result set.",
                "hint": (
                    "Restart discovery without a cursor, or reuse the same query, "
                    "domain and limit that produced "
                    "next_cursor."
                ),
            },
        }
    return _compact_operation_cards_response(
        {
            "kind": "discovery",
            **discovered.model_dump(mode="json"),
            "catalog_resource": "operation://catalog",
            "response_byte_limit": _DISCOVERY_RESPONSE_BYTE_LIMIT,
            "truncation_reason": None,
            "query_metadata_truncated": False,
            "match_metadata_truncated": False,
        },
        cards_key="matches",
        metadata_truncated_key="match_metadata_truncated",
    )


def _operation_browse_response(
    catalog: Catalog,
    *,
    domain: str | None,
    limit: int | None,
    cursor: str | None,
) -> dict[str, Any]:
    try:
        browsed = catalog.browse(
            domain=domain,
            limit=limit if limit is not None else 20,
            cursor=cursor,
        )
    except OperationDiscoveryCursorError:
        return {
            "kind": "error",
            "error": {
                "code": "INVALID_CURSOR",
                "stage": "operation_discovery",
                "message": "The operation browse cursor is not in this result set.",
                "hint": (
                    "Restart browsing without a cursor, or reuse the same domain "
                    "and limit that produced next_cursor."
                ),
            },
        }
    return _compact_operation_cards_response(
        {
            "kind": "browse",
            **browsed.model_dump(mode="json"),
            "catalog_resource": "operation://catalog",
            "response_byte_limit": _DISCOVERY_RESPONSE_BYTE_LIMIT,
            "truncation_reason": None,
            "operation_metadata_truncated": False,
        },
        cards_key="operations",
        metadata_truncated_key="operation_metadata_truncated",
    )


def _compact_operation_cards_response(
    response: dict[str, Any],
    *,
    cards_key: str,
    metadata_truncated_key: str,
) -> dict[str, Any]:
    """Bound one compact response while preserving deterministic pagination."""

    cards = cast(list[dict[str, Any]], response[cards_key])
    while (
        len(_mcp_text_json_bytes(response)) > _DISCOVERY_RESPONSE_BYTE_LIMIT
        and len(cards) > 1
    ):
        cards.pop()
        response["truncated"] = True
        response["next_cursor"] = cards[-1]["operation_id"]
        response["truncation_reason"] = "BYTE_LIMIT"

    if len(_mcp_text_json_bytes(response)) > _DISCOVERY_RESPONSE_BYTE_LIMIT:
        for card in cards:
            if card.get("tags"):
                card["tags"] = []
                response[metadata_truncated_key] = True
                response["truncation_reason"] = "BYTE_LIMIT"

    if len(_mcp_text_json_bytes(response)) > _DISCOVERY_RESPONSE_BYTE_LIMIT:
        for card in cards:
            description = card.get("description")
            if not isinstance(description, str):
                continue
            response[metadata_truncated_key] = True
            response["truncation_reason"] = "BYTE_LIMIT"
            _truncate_text_field_to_budget(
                response,
                card,
                field="description",
                value=description,
            )

    if len(_mcp_text_json_bytes(response)) > _DISCOVERY_RESPONSE_BYTE_LIMIT:
        query = response.get("query")
        if isinstance(query, str):
            response["query_metadata_truncated"] = True
            response["truncation_reason"] = "BYTE_LIMIT"
            _truncate_text_field_to_budget(
                response,
                response,
                field="query",
                value=query,
            )

    if len(_mcp_text_json_bytes(response)) > _DISCOVERY_RESPONSE_BYTE_LIMIT:
        raise RuntimeError("compact operation response exceeds its hard byte limit")
    return response


def _truncate_text_field_to_budget(
    response: dict[str, Any],
    projection: dict[str, Any],
    *,
    field: str,
    value: str,
) -> None:
    """Retain the longest UTF-8 text prefix that fits the response."""

    projection[field] = "."
    if len(_mcp_text_json_bytes(response)) > _DISCOVERY_RESPONSE_BYTE_LIMIT:
        return
    lower = 0
    upper = len(value)
    while lower < upper:
        midpoint = (lower + upper + 1) // 2
        suffix = "..." if midpoint < len(value) else ""
        projection[field] = f"{value[:midpoint]}{suffix}"
        if len(_mcp_text_json_bytes(response)) <= _DISCOVERY_RESPONSE_BYTE_LIMIT:
            lower = midpoint
        else:
            upper = midpoint - 1
    suffix = "..." if lower < len(value) else ""
    projection[field] = f"{value[:lower]}{suffix}" or "."
