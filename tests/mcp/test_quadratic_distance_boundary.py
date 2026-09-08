"""Quadratic profile values compose unchanged through the existing MCP IDs."""

import asyncio

from jacobian.math.geometry.exact._tools import EQUILATERAL_QUADRATIC
from jacobian.mcp.server import create_server
from mcp import Client


def test_quadratic_pair_profile_composes_with_distance_graph() -> None:
    async def scenario() -> None:
        async with Client(create_server(), raise_exceptions=False) as client:
            profile = await client.call_tool(
                "math.run",
                {
                    "operation_id": "geometry.points.distance_profile.compute",
                    "payload": EQUILATERAL_QUADRATIC,
                },
            )
            assert not profile.is_error
            assert profile.structured_content is not None
            result = profile.structured_content["output"]
            assert result["entries"][0]["pairs"] == [[0, 1], [0, 2], [1, 2]]
            selected = await client.call_tool(
                "math.run",
                {
                    "operation_id": "geometry.points.distance_graph.compute",
                    "payload": {
                        "configuration": result["configuration"],
                        "target_squared_distance": result["entries"][0][
                            "squared_distance"
                        ],
                    },
                },
            )
            assert not selected.is_error
            assert selected.structured_content is not None
            assert selected.structured_content["output"]["graph"]["edges"] == [
                [0, 1],
                [0, 2],
                [1, 2],
            ]

    asyncio.run(scenario())
