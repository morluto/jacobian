from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from jacobian.catalog.models import OperationDiscoveryResult
from jacobian.mcp.models import (
    OperationBrowseRequest,
    OperationSearchRequest,
)
from jacobian.mcp.runtime import AppState
from jacobian.mcp.tools import math_find


class _Catalog:
    def search(self, request: Any) -> OperationDiscoveryResult:
        return OperationDiscoveryResult(
            query=request.query,
            matches=(),
            total_matches=0,
            truncated=False,
        )

    def inspect(self, operation_id: str) -> None:
        return None

    def browse(self, **_: Any) -> Any:
        from jacobian.catalog.models import OperationBrowseResult

        return OperationBrowseResult(
            operations=(),
            total_operations=0,
            truncated=False,
        )


def test_math_find_does_not_acquire_an_execution_runtime() -> None:
    state = AppState(
        operation_catalog=_Catalog(),
    )
    context = SimpleNamespace(request_context=SimpleNamespace(lifespan_context=state))

    result = math_find(
        OperationSearchRequest(op="search", query="gcd"),
        ctx=cast(Any, context),
    )

    assert result.root.kind == "discovery"


def test_math_find_search_accepts_a_long_mathematical_query() -> None:
    request = OperationSearchRequest(op="search", query="q" * 513)

    assert len(request.query) == 513


def test_math_find_browse_does_not_acquire_an_execution_runtime() -> None:
    state = AppState(
        operation_catalog=_Catalog(),
    )
    context = SimpleNamespace(request_context=SimpleNamespace(lifespan_context=state))

    result = math_find(
        OperationBrowseRequest(op="browse"),
        ctx=cast(Any, context),
    )

    assert result.root.kind == "browse"
