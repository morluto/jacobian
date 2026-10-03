"""Public composition admits exact used powers and preserves typed refusals."""

import asyncio
import json
from fractions import Fraction
from typing import Any

from mcp.types import TextContent

from jacobian.math.polynomials.series._models import (
    SeriesComposeResult,
    SeriesTruncateResult,
)
from jacobian.mcp.server import create_server
from mcp import Client


def polynomial(values: list[Fraction | int], order: int = 32) -> dict[str, Any]:
    coeffs = [Fraction(v) for v in values] + [Fraction()] * (order - len(values))
    return {
        "variable": "q",
        "truncation_order": order,
        "coefficients": [
            {"num": str(v.numerator), "den": str(v.denominator)} for v in coeffs
        ],
    }


def payload(high_degree: bool = False) -> dict[str, Any]:
    return {
        "outer": polynomial([*([0] * 31), 1] if high_degree else [3, 2, -1]),
        "inner": polynomial([0, Fraction(10**200, 3), Fraction(10**200, 7)]),
    }


def test_mcp_composition_reuses_output_and_preserves_real_growth_refusal() -> None:
    async def scenario() -> None:
        async with Client(create_server(), raise_exceptions=False) as client:
            response = await client.call_tool(
                "math.run",
                {
                    "operation_id": "formal_series.rational.compose.compute",
                    "payload": payload(),
                },
            )
            assert not response.is_error and response.structured_content is not None
            output = response.structured_content["output"]
            result = SeriesComposeResult.model_validate_json(json.dumps(output))
            a, b = Fraction(10**200, 3), Fraction(10**200, 7)
            assert [v.as_fraction() for v in result.result.coefficients] == [
                3,
                2 * a,
                2 * b - a * a,
                -2 * a * b,
                -b * b,
                *([0] * 27),
            ]
            shortened = await client.call_tool(
                "math.run",
                {
                    "operation_id": "formal_series.rational.truncate.compute",
                    "payload": {"series": output["result"], "target_order": 16},
                },
            )
            assert not shortened.is_error and shortened.structured_content is not None
            prefix = SeriesTruncateResult.model_validate_json(
                json.dumps(shortened.structured_content["output"])
            )
            assert prefix.result.coefficients == result.result.coefficients[:16]
            assert prefix.result.variable == "q"
            rejected = await client.call_tool(
                "math.run",
                {
                    "operation_id": "formal_series.rational.compose.compute",
                    "payload": payload(True),
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
                    "operation_id": "formal_series.rational.compose.compute",
                    "payload": payload(),
                },
            )
            assert not recovered.is_error and recovered.structured_content is not None
            assert recovered.structured_content["output"] == output

    asyncio.run(scenario())
