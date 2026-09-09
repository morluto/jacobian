"""Bounded search projection and cancellation helpers for the MCP adapter."""

from __future__ import annotations

from typing import Literal

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationMatchRequest
from jacobian.catalog.search import OperationDiscoveryCursorError
from jacobian.mcp.models import (
    OperationDiscoveryError,
    OperationDiscoveryErrorDetail,
    OperationFindResult,
)


def _operation_match_response(
    catalog: Catalog,
    *,
    need: str,
    namespace: str | None,
    limit: int | None,
    cursor: str | None,
    search_mode: Literal["precise", "broad"] = "precise",
) -> OperationFindResult | OperationDiscoveryError:
    match_request = OperationMatchRequest(
        need=need,
        namespace=namespace,
        limit=limit if limit is not None else 5,
        cursor=cursor,
        search_mode=search_mode,
    )
    try:
        matched = catalog.match(match_request)
    except OperationDiscoveryCursorError:
        return OperationDiscoveryError(
            kind="error",
            error=OperationDiscoveryErrorDetail(
                code="INVALID_CURSOR",
                stage="operation_discovery",
                message="The operation discovery cursor is not in this result set.",
                hint=(
                    "Restart matching without a cursor, or reuse the same need and "
                    "namespace that produced next_cursor. The limit may change."
                ),
            ),
        )
    return OperationFindResult(
        kind="matches",
        need=matched.need,
        namespace=matched.namespace,
        search_mode=matched.search_mode,
        matches=matched.matches,
        total_matches=matched.total_matches,
        next_cursor=matched.next_cursor,
    )
