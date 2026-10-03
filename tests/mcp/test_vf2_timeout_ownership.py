"""Live SDK projections retain VF2's limiting owner through dispatch."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from mcp.types import TextContent
from tests.support.vf2_timeout import VF2_PAYLOAD, TimeoutPath, patch_vf2_timeout

from jacobian._execution import (
    bind_request_deadline,
    current_request_execution,
    request_execution,
)
from jacobian.catalog.catalog import Catalog
from jacobian.math.graphs.isomorphism._tools import TOOLS
from jacobian.mcp import direct_tools, tools
from jacobian.mcp.direct_tools import direct_operation_tools
from jacobian.mcp.runtime import AppState
from jacobian.mcp.server import _build_server
from mcp import Client


@pytest.mark.parametrize("direct", [False, True], ids=["math-run", "direct"])
@pytest.mark.parametrize("path", ["reserve", "prelaunch", "worker", "delivery"])
@pytest.mark.parametrize(
    ("outer", "bound", "expected_owner", "source_owns"),
    [
        (190.0, None, "operation_wall", True),
        (130.0, None, "caller_deadline", False),
        (None, 130.0, "operation_wall", False),
        (160.0, None, "caller_deadline", False),
    ],
    ids=["source", "caller", "prior-operation", "source-tie"],
)
def test_vf2_timeout_owner_through_sdk(
    monkeypatch: pytest.MonkeyPatch,
    direct: bool,
    path: TimeoutPath,
    outer: float | None,
    bound: float | None,
    expected_owner: str,
    source_owns: bool,
) -> None:
    operation = TOOLS[0]
    catalog = Catalog((operation,))
    adapter = direct_tools if direct else tools
    execute = adapter.execute_operation
    traces: list[list[str]] = []

    def enclosing_execution(*args: Any, **kwargs: Any) -> Any:
        cutoff = min(value for value in (160.0, outer, bound) if value is not None)
        clock = [cutoff - 0.001 if path == "reserve" else 100.0]
        with monkeypatch.context() as synchronous_patch:
            traces.append(patch_vf2_timeout(synchronous_patch, clock, path))
            with request_execution(100.0, outer_deadline=outer) as parent:
                if bound is not None:
                    bind_request_deadline(bound)
                inherited_deadline = parent.deadline
                try:
                    return execute(*args, **kwargs)
                finally:
                    assert current_request_execution() is parent
                    assert parent.deadline == inherited_deadline

    monkeypatch.setattr(adapter, "execute_operation", enclosing_execution)
    server = _build_server(
        state=AppState(operation_catalog=catalog),
        evaluation_tools=direct_operation_tools(catalog) if direct else (),
    )

    async def scenario() -> None:
        async with Client(server, raise_exceptions=False) as client:
            response = await client.call_tool(
                operation.operation_id if direct else "math.run",
                VF2_PAYLOAD
                if direct
                else {"operation_id": operation.operation_id, "payload": VF2_PAYLOAD},
            )
        assert response.is_error
        content = response.content[0]
        assert isinstance(content, TextContent)
        diagnostic = json.loads(content.text[content.text.index("{") :])
        assert diagnostic["code"] == "OPERATION_TIMEOUT"
        assert diagnostic["timeout_owner"] == expected_owner
        assert diagnostic["stage"] == "operation_execution"
        if source_owns and path in {"reserve", "prelaunch"}:
            assert diagnostic["configured_seconds"] == 60
        else:
            assert "configured_seconds" not in diagnostic
        assert "adjustable_field_path" not in diagnostic
        assert current_request_execution() is None

    asyncio.run(scenario())
    expected_trace = (
        ["lease", "worker", "delivery"]
        if path == "delivery"
        else ["lease", "worker"]
        if path == "worker"
        else ["lease"]
    )
    assert traces == [expected_trace]
