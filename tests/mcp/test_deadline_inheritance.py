"""Both real SDK entry points retain an adapter's enclosing timing context."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import pytest
from mcp.types import TextContent

from jacobian._execution import (
    bind_request_deadline,
    current_request_execution,
    request_execution,
)
from jacobian.catalog.catalog import Catalog
from jacobian.math.graphs.optimization._chromatic_bipartition import (
    CHROMATIC_BIPARTITION_OPERATION,
)
from jacobian.mcp import direct_tools, tools
from jacobian.mcp.direct_tools import direct_operation_tools
from jacobian.mcp.runtime import AppState
from jacobian.mcp.server import _build_server
from mcp import Client


@pytest.mark.parametrize("direct", [False, True], ids=["math-run", "direct"])
@pytest.mark.parametrize(
    "timing",
    [
        "expired-outer",
        "expired-bound",
        "past-origin",
        "live-outer",
        "live-bound",
        "none",
    ],
)
def test_mcp_inherits_adapter_timing(
    monkeypatch: pytest.MonkeyPatch, direct: bool, timing: str
) -> None:
    catalog = Catalog((CHROMATIC_BIPARTITION_OPERATION,))
    adapter = direct_tools if direct else tools
    execute = adapter.execute_operation
    restored: list[bool] = []

    def enclosing_execution(*args: Any, **kwargs: Any) -> Any:
        if timing == "none":
            return execute(*args, **kwargs)
        now = time.monotonic()
        deadline = now - 1 if timing.startswith("expired") else now + 60
        with request_execution(
            now - (130 if timing == "past-origin" else 10),
            outer_deadline=deadline if timing.endswith("outer") else None,
        ) as parent:
            if timing.endswith("bound"):
                bind_request_deadline(deadline)
            expected_deadline = parent.deadline
            try:
                return execute(*args, **kwargs)
            finally:
                assert current_request_execution() is parent
                assert parent.deadline == expected_deadline
                restored.append(True)

    # The wire has no deadline field. Model the enclosing serving context at
    # the synchronous adapter seam actually used by each SDK entry point.
    monkeypatch.setattr(adapter, "execute_operation", enclosing_execution)
    server = _build_server(
        state=AppState(operation_catalog=catalog),
        evaluation_tools=direct_operation_tools(catalog) if direct else (),
    )
    payload = {
        "graph": {"vertices": [], "edges": []},
        "s": 1,
        "t": 1,
        "resource_budget": {"wall_seconds": 120},
    }
    operation_id = CHROMATIC_BIPARTITION_OPERATION.operation_id

    async def scenario() -> None:
        async with Client(server, raise_exceptions=False) as client:
            response = await client.call_tool(
                operation_id if direct else "math.run",
                payload
                if direct
                else {"operation_id": operation_id, "payload": payload},
            )
        if timing.startswith("expired") or timing == "past-origin":
            assert response.is_error
            content = response.content[0]
            assert isinstance(content, TextContent)
            diagnostic = json.loads(content.text[content.text.index("{") :])
            assert diagnostic["code"] == "OPERATION_TIMEOUT"
            assert diagnostic["operation_id"] == operation_id
            assert diagnostic["timeout_owner"] == (
                "caller_deadline" if timing == "expired-outer" else "operation_wall"
            )
            assert diagnostic["stage"] == (
                "operation_execution" if timing == "past-origin" else "request_parsing"
            )
        else:
            assert not response.is_error
            assert response.structured_content is not None
            output = (
                response.structured_content
                if direct
                else response.structured_content["output"]
            )
            assert output["status"] == "NO_SPLIT"
            assert output["checked_partitions"] == 0
        assert current_request_execution() is None

    asyncio.run(scenario())
    assert restored == ([] if timing == "none" else [True])
