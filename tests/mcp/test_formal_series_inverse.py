"""Live inverse admission and recovery through the MCP SDK boundary."""

import asyncio

import pytest
from mcp.shared.exceptions import MCPError

from jacobian.math.polynomials.series import inverse
from jacobian.math.polynomials.series._models import TruncatedSeries
from jacobian.mcp.server import create_server
from mcp import Client


def test_inverse_native_mcp_parity_after_growth_rejection() -> None:
    async def scenario() -> None:
        async with Client(create_server(), raise_exceptions=True) as client:
            with pytest.raises(MCPError) as error:
                await client.call_tool(
                    "math.run",
                    {
                        "operation_id": "formal_series.rational.inverse.compute",
                        "payload": {
                            "variable": "x",
                            "truncation_order": 20,
                            "coefficients": [
                                {"num": "1", "den": "1"},
                                {"num": str(10**255), "den": "1"},
                                *[{"num": "0", "den": "1"}] * 18,
                            ],
                        },
                    },
                )
            assert error.value.code == -32602
            source = TruncatedSeries.model_validate(
                {
                    "variable": "x",
                    "truncation_order": 6,
                    "coefficients": [{"num": "1", "den": "1"}] * 2
                    + [{"num": "0", "den": "1"}] * 4,
                }
            )
            result = await client.call_tool(
                "math.run",
                {
                    "operation_id": "formal_series.rational.inverse.compute",
                    "payload": source.model_dump(mode="json"),
                },
            )
            assert not result.is_error
            assert result.structured_content is not None
            assert result.structured_content["output"] == inverse(source).model_dump(
                mode="json"
            )

    asyncio.run(scenario())
