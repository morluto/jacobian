from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationMatchResult
from jacobian.mcp.models import OperationInspectRequest, OperationMatchRequest
from jacobian.mcp.runtime import AppState
from jacobian.mcp.tools import math_find


class _Catalog:
    def match(self, request: Any) -> OperationMatchResult:
        return OperationMatchResult(
            need=request.need,
            matches=(),
            total_matches=0,
        )

    def inspect(self, operation_id: str) -> None:
        return None


def test_math_find_does_not_acquire_an_execution_runtime() -> None:
    state = AppState(
        operation_catalog=cast(Catalog, _Catalog()),
    )
    context = SimpleNamespace(request_context=SimpleNamespace(lifespan_context=state))

    result = math_find(
        OperationMatchRequest(
            op="match", need="compute an exact greatest common divisor"
        ),
        ctx=cast(Any, context),
    )

    assert result.root.kind == "matches"


def test_math_find_inspection_does_not_acquire_an_execution_runtime() -> None:
    state = AppState(
        operation_catalog=cast(Catalog, _Catalog()),
    )
    context = SimpleNamespace(request_context=SimpleNamespace(lifespan_context=state))

    result = math_find(
        OperationInspectRequest(op="inspect", operation_id="integer.compute.unknown"),
        ctx=cast(Any, context),
    )

    assert result.root.kind == "error"
