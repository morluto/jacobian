"""Projective operation errors remain classified through the MCP boundary."""

import asyncio

from mcp.types import TextContent

from jacobian.mcp.server import create_server
from mcp import Client


def test_projective_composition_base_locus_is_owner_error_over_mcp() -> None:
    async def scenario() -> None:
        payload = {
            "outer": {
                "degree": 1,
                "numerator": [{"num": "0", "den": "1"}, {"num": "1", "den": "1"}],
                "denominator": [{"num": "0", "den": "1"}, {"num": "1", "den": "1"}],
            },
            "inner": {
                "degree": 1,
                "numerator": [{"num": "0", "den": "1"}, {"num": "0", "den": "1"}],
                "denominator": [{"num": "0", "den": "1"}, {"num": "1", "den": "1"}],
            },
        }
        async with Client(create_server(), raise_exceptions=False) as client:
            result = await client.call_tool(
                "math.run",
                {
                    "operation_id": "arithmetic_dynamics.projective_map.compose.compute",
                    "payload": payload,
                },
            )
        assert result.is_error
        assert result.structured_content is None
        assert isinstance(result.content[0], TextContent)
        assert "arithmetic_dynamics.projective_base_locus" in result.content[0].text

    asyncio.run(scenario())
