"""Complete power results survive public execution and unchanged handoffs."""

import asyncio
import json
from typing import Any

from mcp.types import TextContent

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.polynomials.series import power
from jacobian.math.polynomials.series._models import (
    SeriesPowerRequest,
    SeriesPowerResult,
    SeriesTruncateResult,
)
from jacobian.mcp.server import create_server
from mcp import Client


def payload(order: int, exponent: int = 1000) -> dict[str, Any]:
    return {
        "series": {
            "variable": "x",
            "truncation_order": order,
            "coefficients": [
                {"num": str(int(i < 3)), "den": "1"} for i in range(order)
            ],
        },
        "exponent": exponent,
    }


def test_public_complete_distribution_matches_native() -> None:
    encoded = payload(2001)
    request = SeriesPowerRequest.model_validate_json(json.dumps(encoded))
    native = power(request.series, request.exponent)
    public = invoke_operation(
        "formal_series.rational.power.compute", encoded, Catalog.open()
    )
    assert public.output == native.model_dump(mode="json")
    result = SeriesPowerResult.model_validate_json(json.dumps(public.output))
    assert len(result.result.coefficients) == 2001
    assert sum(v.as_fraction() for v in result.result.coefficients) == 3**1000
    assert (
        result.result.coefficients[0].as_fraction()
        == result.result.coefficients[-1].as_fraction()
        == 1
    )


def test_mcp_power_output_handoff_refusal_and_recovery() -> None:
    async def scenario() -> None:
        async with Client(create_server(), raise_exceptions=False) as client:
            response = await client.call_tool(
                "math.run",
                {
                    "operation_id": "formal_series.rational.power.compute",
                    "payload": payload(2001),
                },
            )
            assert not response.is_error and response.structured_content is not None
            output = response.structured_content["output"]
            result = SeriesPowerResult.model_validate_json(json.dumps(output))
            assert len(result.result.coefficients) == 2001
            assert sum(v.as_fraction() for v in result.result.coefficients) == 3**1000
            prefix = await client.call_tool(
                "math.run",
                {
                    "operation_id": "formal_series.rational.truncate.compute",
                    "payload": {"series": output["result"], "target_order": 512},
                },
            )
            assert not prefix.is_error and prefix.structured_content is not None
            decoded = SeriesTruncateResult.model_validate_json(
                json.dumps(prefix.structured_content["output"])
            )
            assert decoded.result.coefficients == result.result.coefficients[:512]
            assert decoded.result.variable == result.result.variable
            refused = await client.call_tool(
                "math.run",
                {
                    "operation_id": "formal_series.rational.power.compute",
                    "payload": payload(2049, 2),
                },
            )
            assert refused.is_error
            text = refused.content[0]
            assert isinstance(text, TextContent)
            diagnostic = json.loads(
                text.text.removeprefix("Error executing tool math.run: ")
            )
            assert diagnostic["code"] == "RESOURCE_ADMISSION_REJECTED"
            assert diagnostic["stage"] == "resource_admission"
            recovered = await client.call_tool(
                "math.run",
                {
                    "operation_id": "formal_series.rational.power.compute",
                    "payload": payload(3, 0),
                },
            )
            assert not recovered.is_error and recovered.structured_content is not None
            unit = SeriesPowerResult.model_validate_json(
                json.dumps(recovered.structured_content["output"])
            )
            assert [v.as_fraction() for v in unit.result.coefficients] == [1, 0, 0]
            assert unit.multiplication_count == 0

    asyncio.run(scenario())
