"""Both SDK entry points retain source-authored context from a real worker."""

import asyncio
import json
import logging
from typing import Any

import pytest
from mcp.types import TextContent
from tests.support.chromatic_timeout import chromatic_worker_timeout

from jacobian.catalog.catalog import Catalog
from jacobian.math.graphs.optimization import (
    _chromatic_bipartition_process as process_owner,
)
from jacobian.math.graphs.optimization._chromatic_bipartition import (
    CHROMATIC_BIPARTITION_OPERATION,
)
from jacobian.mcp.direct_tools import direct_operation_tools
from jacobian.mcp.runtime import AppState
from jacobian.mcp.server import _build_server
from jacobian.process import BoundedProcessResult
from mcp import Client


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("wall_seconds", [5, 120])
def test_worker_timeout_context_through_sdk(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    direct: bool,
    wall_seconds: int,
) -> None:
    request, output = chromatic_worker_timeout(monkeypatch, wall_seconds)
    monkeypatch.setattr(
        process_owner,
        "run_bounded_process",
        lambda *args, **kwargs: BoundedProcessResult(
            0, output, b"", False, False, False
        ),
    )
    operation = CHROMATIC_BIPARTITION_OPERATION
    catalog = Catalog((operation,))
    server = _build_server(
        state=AppState(operation_catalog=catalog),
        evaluation_tools=direct_operation_tools(catalog) if direct else (),
    )
    payload = request.model_dump(mode="json")

    async def scenario() -> Any:
        async with Client(server, raise_exceptions=False) as client:
            return await client.call_tool(
                operation.operation_id if direct else "math.run",
                payload
                if direct
                else {"operation_id": operation.operation_id, "payload": payload},
            )

    with caplog.at_level(logging.ERROR, logger="jacobian.mcp.tools"):
        result = asyncio.run(scenario())
    assert result.is_error
    assert result.structured_content is None
    assert isinstance(result.content[0], TextContent)
    text = result.content[0].text
    assert json.loads(text[text.index("{") :]) == {
        "code": "OPERATION_TIMEOUT",
        "message": "operation deadline expired",
        "operation_id": operation.operation_id,
        "stage": "operation_execution",
        "timeout_owner": "operation_wall",
        "configured_seconds": wall_seconds,
        "adjustable_field_path": ["resource_budget", "wall_seconds"],
        "deterministic_work_remains_fixed": False,
    }
    assert not [
        record
        for record in caplog.records
        if record.name == "jacobian.mcp.tools" and record.exc_info
    ]
