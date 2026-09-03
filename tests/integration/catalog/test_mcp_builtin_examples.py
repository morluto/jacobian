"""MCP projection of unexpected faults from catalog operations.

Advertised examples are exhaustively executed at their dispatch boundary in
``test_builtin_examples.py``. The generic successful MCP projection is owned by
``tests/mcp/test_mcp_invocation_journey.py``; repeating every mathematical
kernel through that same projection here would duplicate both contracts.
"""

from __future__ import annotations

import asyncio

from mcp.types import TextContent

from jacobian.catalog.catalog import Catalog
from mcp import Client


def test_mcp_unexpected_operation_fault_uses_the_sdk_failure_path() -> None:
    """Unexpected operation faults are not recast as mathematical tool errors."""

    from pydantic import Field

    from jacobian._models import StrictModel
    from jacobian.catalog.builtins import BUILTIN_TOOLS
    from jacobian.catalog.models import MathTool

    class BoomRequest(StrictModel):
        x: int = Field(ge=0)

    class BoomResult(StrictModel):
        y: int

    def boom(_request: BoomRequest) -> BoomResult:
        raise RuntimeError("synthetic boom")

    boom_tool = MathTool(
        operation_id="test.synthetic.boom",
        title="Synthetic boom",
        description="Always raises.",
        request_type=BoomRequest,
        result_type=BoomResult,
        run=boom,
    )

    catalog = Catalog((*BUILTIN_TOOLS, boom_tool))

    from jacobian.mcp.runtime import AppState
    from jacobian.mcp.server import _build_server

    server = _build_server(state=AppState(operation_catalog=catalog))

    async def scenario() -> None:
        async with Client(server, raise_exceptions=False) as client:
            result = await client.call_tool(
                "math.run",
                {"operation_id": "test.synthetic.boom", "payload": {"x": 1}},
            )
            # Expected mathematical failures are returned by their owning operation.
            # A programming fault reaches the MCP SDK's installed failure path rather
            # than being mislabeled as a generic Jacobian ToolError.
            assert result.is_error is True
            assert result.structured_content is None
            first = result.content[0] if result.content else None
            text = first.text if isinstance(first, TextContent) else ""
            assert text == "Error executing tool math.run: operation execution failed"
            assert len(text) < 5000

    asyncio.run(scenario())
