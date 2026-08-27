"""Bounded search projection and cancellation helpers for the MCP adapter."""

from __future__ import annotations

from typing import Any

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDiscoveryRequest
from jacobian.catalog.search import OperationDiscoveryCursorError


def _operation_discovery_response(
    catalog: Catalog,
    *,
    query: str,
    namespace: str | None,
    limit: int | None,
    cursor: str | None,
) -> dict[str, Any]:
    discovery_request = OperationDiscoveryRequest(
        query=query,
        namespace=namespace,
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
                    "namespace and limit that produced "
                    "next_cursor."
                ),
            },
        }
    return {
        "kind": "discovery",
        **discovered.model_dump(mode="json"),
        "catalog_resource": "operation://catalog",
    }


def _operation_browse_response(
    catalog: Catalog,
    *,
    namespace: str | None,
    limit: int | None,
    cursor: str | None,
) -> dict[str, Any]:
    try:
        browsed = catalog.browse(
            namespace=namespace,
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
                    "Restart browsing without a cursor, or reuse the same namespace "
                    "and limit that produced next_cursor."
                ),
            },
        }
    return {
        "kind": "browse",
        **browsed.model_dump(mode="json"),
        "catalog_resource": "operation://catalog",
    }
