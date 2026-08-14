from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from jacobian.adapters.mcp.context import AppState
from jacobian.adapters.mcp.tools import math_find
from jacobian.contracts.operation_find import OperationSearchRequest
from jacobian.contracts.operations import OperationDiscoveryResult


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
