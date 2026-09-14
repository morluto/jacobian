"""MCP construction and direct hypergraph composition for sunflowers."""

import asyncio
import json

from jacobian.math.combinatorics.extremal_sets._sunflower_r import (
    SunflowerFamilyResult,
)
from jacobian.mcp.server import create_server
from mcp import Client


def test_sunflower_family_mcp_round_trips_and_composes() -> None:
    async def scenario() -> None:
        payload = {
            "source": {
                "ground_set_size": 6,
                "members": [[0, 1], [0, 2], [0, 4], [0, 5], [1, 2], [4, 5]],
            },
            "petal_count": 3,
        }
        async with Client(create_server(), raise_exceptions=False) as client:
            response = await client.call_tool(
                "math.run",
                {
                    "operation_id": "set_system.sunflower_family.construct",
                    "payload": payload,
                },
            )
            assert not response.is_error
            assert response.structured_content is not None
            output = response.structured_content["output"]
            decoded = SunflowerFamilyResult.model_validate_json(json.dumps(output))
            assert decoded.sunflower_count == 4
            assert decoded.sunflower_free is False

            parameters = await client.call_tool(
                "math.run",
                {
                    "operation_id": "hypergraph.parameters.compute",
                    "payload": {"hypergraph": output["hypergraph"]},
                },
            )
            assert not parameters.is_error
            assert parameters.structured_content is not None
            assert parameters.structured_content["output"]["edge_count"] == 4
            assert parameters.structured_content["output"]["uniform_size"] == 3

    asyncio.run(scenario())
