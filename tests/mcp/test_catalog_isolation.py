from __future__ import annotations

import hashlib
import hmac
import logging
from types import SimpleNamespace
from typing import Any, cast

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationMatchResult
from jacobian.mcp import tools
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


def test_math_find_logs_a_hashed_match_query(
    caplog: pytest.LogCaptureFixture,
) -> None:
    state = AppState(
        operation_catalog=cast(Catalog, _Catalog()),
    )
    context = SimpleNamespace(request_context=SimpleNamespace(lifespan_context=state))
    need = "find an exact gcd\nwithout logging a run payload"

    with caplog.at_level(logging.INFO, logger="jacobian.mcp.tools"):
        math_find(
            OperationMatchRequest(
                op="match",
                need=need,
            ),
            ctx=cast(Any, context),
        )

    records = [
        record for record in caplog.records if record.name == "jacobian.mcp.tools"
    ]
    assert len(records) == 1
    message = records[0].getMessage()
    query_hash = hmac.new(
        tools._FIND_QUERY_LOG_KEY, need.encode("utf-8"), hashlib.sha256
    ).hexdigest()[:16]
    assert message == f"math.find query_hash={query_hash}"
    assert need not in message
    assert "find an exact gcd" not in message
    assert "query='" not in message
    assert "\n" not in message
