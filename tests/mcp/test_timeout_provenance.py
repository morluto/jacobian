"""Both MCP entry points retain checkpoint ownership and sanitized diagnostics."""

import asyncio
import json
import logging
from types import SimpleNamespace

import pytest
from mcp.types import TextContent

from jacobian import _execution
from jacobian._execution import (
    OperationExecutionStage,
    TimeoutOwner,
    bind_request_deadline,
    request_checkpoint,
    request_execution,
)
from jacobian._models import StrictModel
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import MathTool
from jacobian.mcp.direct_tools import direct_operation_tools
from jacobian.mcp.runtime import AppState
from jacobian.mcp.server import _build_server
from mcp import Client


class _Value(StrictModel):
    value: int


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize(
    ("operation_deadline", "outer_deadline", "expected_owner"),
    [
        (105.0, 110.0, TimeoutOwner.OPERATION_WALL),
        (110.0, 105.0, TimeoutOwner.CALLER_DEADLINE),
        (105.0, 105.0, TimeoutOwner.CALLER_DEADLINE),
    ],
)
def test_sdk_timeout_preserves_the_limiting_deadline_owner(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    direct: bool,
    operation_deadline: float,
    outer_deadline: float,
    expected_owner: TimeoutOwner,
) -> None:
    monkeypatch.setattr(_execution, "time", SimpleNamespace(monotonic=lambda: 111.0))

    def run(request: _Value) -> _Value:
        with request_execution(100.0, outer_deadline=outer_deadline):
            bind_request_deadline(operation_deadline)
            request_checkpoint(
                "private checkpoint marker",
                public_stage=OperationExecutionStage.RESULT_PROJECTION,
            )
        return request

    operation = MathTool(
        operation_id="test.timeout.provenance",
        title="Timeout provenance sentinel",
        description="Exercises checkpoint ownership through both MCP entry points.",
        request_type=_Value,
        result_type=_Value,
        run=run,
    )
    catalog = Catalog((operation,))
    server = _build_server(
        state=AppState(operation_catalog=catalog),
        evaluation_tools=direct_operation_tools(catalog) if direct else (),
    )

    async def scenario() -> None:
        async with Client(server, raise_exceptions=False) as client:
            result = await client.call_tool(
                operation.operation_id if direct else "math.run",
                {"value": 7}
                if direct
                else {"operation_id": operation.operation_id, "payload": {"value": 7}},
            )

        assert result.is_error
        assert result.structured_content is None
        assert isinstance(result.content[0], TextContent)
        text = result.content[0].text
        diagnostic = json.loads(text[text.index("{") :])
        assert diagnostic == {
            "code": "OPERATION_TIMEOUT",
            "operation_id": operation.operation_id,
            "stage": "result_projection",
            "message": "operation deadline expired",
            "timeout_owner": expected_owner.value,
            "elapsed_seconds": 11.0,
            "deterministic_work_remains_fixed": False,
        }
        assert "private checkpoint marker" not in text

    with caplog.at_level(logging.ERROR, logger="jacobian.mcp.tools"):
        asyncio.run(scenario())
    assert not [
        record
        for record in caplog.records
        if record.name == "jacobian.mcp.tools" and record.exc_info
    ]
