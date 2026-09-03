from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, cast

import pytest

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


def test_math_find_inspection_does_not_log_a_query_or_acquire_a_runtime(
    caplog: pytest.LogCaptureFixture,
) -> None:
    state = AppState(
        operation_catalog=cast(Catalog, _Catalog()),
    )
    context = SimpleNamespace(request_context=SimpleNamespace(lifespan_context=state))

    with caplog.at_level(logging.INFO, logger="jacobian.mcp.tools"):
        result = math_find(
            OperationInspectRequest(
                op="inspect", operation_id="integer.compute.unknown"
            ),
            ctx=cast(Any, context),
        )

    assert result.root.kind == "error"
    assert not [
        record for record in caplog.records if record.name == "jacobian.mcp.tools"
    ]


def test_math_find_logs_the_match_query_on_one_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    state = AppState(
        operation_catalog=cast(Catalog, _Catalog()),
    )
    context = SimpleNamespace(request_context=SimpleNamespace(lifespan_context=state))

    with caplog.at_level(logging.INFO, logger="jacobian.mcp.tools"):
        math_find(
            OperationMatchRequest(
                op="match",
                need="find an exact gcd\nwithout logging a run payload",
            ),
            ctx=cast(Any, context),
        )

    records = [
        record for record in caplog.records if record.name == "jacobian.mcp.tools"
    ]
    assert len(records) == 1
    assert records[0].getMessage() == (
        "math.find query='find an exact gcd\\nwithout logging a run payload'"
    )
