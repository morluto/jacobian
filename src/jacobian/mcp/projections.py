"""Bounded search projection and cancellation helpers for the MCP adapter."""

from __future__ import annotations

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDiscoveryRequest
from jacobian.catalog.search import OperationDiscoveryCursorError
from jacobian.mcp.models import (
    OperationBrowseResult,
    OperationDiscoveryError,
    OperationDiscoveryErrorDetail,
    OperationSearchResult,
)


def _operation_discovery_response(
    catalog: Catalog,
    *,
    query: str,
    namespace: str | None,
    limit: int | None,
    cursor: str | None,
) -> OperationSearchResult | OperationDiscoveryError:
    discovery_request = OperationDiscoveryRequest(
        query=query,
        namespace=namespace,
        limit=limit if limit is not None else 5,
        cursor=cursor,
    )
    try:
        discovered = catalog.search(discovery_request)
    except OperationDiscoveryCursorError:
        return OperationDiscoveryError(
            kind="error",
            error=OperationDiscoveryErrorDetail(
                code="INVALID_CURSOR",
                stage="operation_discovery",
                message="The operation discovery cursor is not in this result set.",
                hint=(
                    "Restart discovery without a cursor, or reuse the same query, "
                    "namespace and limit that produced "
                    "next_cursor."
                ),
            ),
        )
    return OperationSearchResult(
        kind="discovery",
        query=discovered.query,
        namespace=discovered.namespace,
        matches=discovered.matches,
        total_matches=discovered.total_matches,
        next_cursor=discovered.next_cursor,
    )


def _operation_browse_response(
    catalog: Catalog,
    *,
    namespace: str | None,
    limit: int | None,
    cursor: str | None,
) -> OperationBrowseResult | OperationDiscoveryError:
    try:
        browsed = catalog.browse(
            namespace=namespace,
            limit=limit if limit is not None else 20,
            cursor=cursor,
        )
    except OperationDiscoveryCursorError:
        return OperationDiscoveryError(
            kind="error",
            error=OperationDiscoveryErrorDetail(
                code="INVALID_CURSOR",
                stage="operation_discovery",
                message="The operation browse cursor is not in this result set.",
                hint=(
                    "Restart browsing without a cursor, or reuse the same namespace "
                    "and limit that produced next_cursor."
                ),
            ),
        )
    return OperationBrowseResult(
        kind="browse",
        namespace=browsed.namespace,
        operations=browsed.operations,
        total_operations=browsed.total_operations,
        next_cursor=browsed.next_cursor,
    )
