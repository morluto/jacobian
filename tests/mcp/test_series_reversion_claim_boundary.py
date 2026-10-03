"""Real MCP reversion delivers usable claims and typed resource refusal."""

import asyncio
import json
from typing import Any

from mcp.types import TextContent

from jacobian.math.polynomials.series import verify_reversion
from jacobian.math.polynomials.series._models import (
    SeriesReversionResult,
    SeriesTruncateResult,
)
from jacobian.mcp.server import create_server
from mcp import Client


def payload(order: int) -> dict[str, Any]:
    return {
        "variable": "q",
        "truncation_order": order,
        "coefficients": [
            {"num": str(v), "den": "1"} for v in [0, 1, 1, *([0] * (order - 3))]
        ],
    }


def test_mcp_reversion_claim_reuse_refusal_and_recovery() -> None:
    async def scenario() -> None:
        async with Client(create_server(), raise_exceptions=False) as client:
            response = await client.call_tool(
                "math.run",
                {
                    "operation_id": "formal_series.rational.reversion.compute",
                    "payload": payload(8),
                },
            )
            assert not response.is_error and response.structured_content is not None
            output = response.structured_content["output"]
            claim = SeriesReversionResult.model_validate_json(json.dumps(output))
            assert verify_reversion(claim)
            assert [v.as_fraction() for v in claim.result.coefficients] == [
                0,
                1,
                -1,
                2,
                -5,
                14,
                -42,
                132,
            ]
            shortened = await client.call_tool(
                "math.run",
                {
                    "operation_id": "formal_series.rational.truncate.compute",
                    "payload": {"series": output["result"], "target_order": 4},
                },
            )
            assert not shortened.is_error and shortened.structured_content is not None
            prefix = SeriesTruncateResult.model_validate_json(
                json.dumps(shortened.structured_content["output"])
            )
            assert prefix.result.coefficients == claim.result.coefficients[:4]
            rejected = await client.call_tool(
                "math.run",
                {
                    "operation_id": "formal_series.rational.reversion.compute",
                    "payload": payload(513),
                },
            )
            assert rejected.is_error
            content = rejected.content[0]
            assert isinstance(content, TextContent)
            diagnostic = json.loads(
                content.text.removeprefix("Error executing tool math.run: ")
            )
            assert diagnostic["code"] == "RESOURCE_ADMISSION_REJECTED"
            assert diagnostic["stage"] == "resource_admission"
            recovered = await client.call_tool(
                "math.run",
                {
                    "operation_id": "formal_series.rational.reversion.compute",
                    "payload": payload(8),
                },
            )
            assert not recovered.is_error and recovered.structured_content is not None
            assert recovered.structured_content["output"] == output

    asyncio.run(scenario())
