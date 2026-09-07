"""A failed min-cost backend cannot establish mathematical infeasibility."""

import asyncio

import networkx as nx
import pytest
from mcp.types import TextContent

from jacobian.mcp.server import create_server
from mcp import Client


def test_min_cost_backend_error_is_mcp_execution_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> object:
        raise nx.NetworkXError("private network backend failure")

    monkeypatch.setattr(nx, "network_simplex", fail)

    async def scenario() -> None:
        async with Client(create_server(), raise_exceptions=False) as client:
            result = await client.call_tool(
                "math.run",
                {
                    "operation_id": "network.min_cost_flow.compute",
                    "payload": {
                        "graph": {
                            "vertex_count": 2,
                            "edges": [
                                {
                                    "source": 0,
                                    "target": 1,
                                    "capacity": {"num": "1", "den": "1"},
                                    "cost": {"num": "1", "den": "1"},
                                }
                            ],
                        },
                        "demands": [-1, 1],
                    },
                },
            )
        assert result.is_error
        assert result.structured_content is None
        assert isinstance(result.content[0], TextContent)
        assert (
            result.content[0].text
            == "Error executing tool math.run: operation execution failed"
        )

    asyncio.run(scenario())
