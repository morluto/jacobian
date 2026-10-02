"""Public unit-series scale: exact output handoff, refusal, and recovery."""

import asyncio
import json
from typing import Any

from mcp.types import TextContent

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.polynomials.series import inverse, verify_divide, verify_inverse
from jacobian.math.polynomials.series._models import (
    SeriesDivideResult,
    SeriesInverseRequest,
    SeriesInverseResult,
)
from jacobian.mcp.server import create_server
from mcp import Client


def _payload(n: int) -> dict[str, Any]:
    return {
        "variable": "q",
        "truncation_order": n,
        "coefficients": [
            {"num": "1", "den": "1"},
            {"num": "-1", "den": "1"},
            *({"num": "0", "den": "1"} for _ in range(n - 2)),
        ],
    }


def test_public_inverse_at_2048_has_canonical_native_identity() -> None:
    payload = _payload(2048)
    request = SeriesInverseRequest.model_validate_json(json.dumps(payload))
    native = inverse(request.as_series())
    public = invoke_operation(
        "formal_series.rational.inverse.compute", payload, Catalog.open()
    )
    assert public.output == native.model_dump(mode="json")
    decoded = SeriesInverseResult.model_validate_json(json.dumps(public.output))
    assert verify_inverse(decoded)
    assert len(decoded.result.coefficients) == 2048


def test_mcp_actual_inverse_output_feeds_division_then_refusal_recovers() -> None:
    async def scenario() -> None:
        async with Client(create_server(), raise_exceptions=False) as client:
            payload = _payload(513)
            response = await client.call_tool(
                "math.run",
                {
                    "operation_id": "formal_series.rational.inverse.compute",
                    "payload": payload,
                },
            )
            assert not response.is_error
            assert response.structured_content is not None
            output = response.structured_content["output"]
            decoded = SeriesInverseResult.model_validate_json(json.dumps(output))
            assert verify_inverse(decoded)
            quotient = await client.call_tool(
                "math.run",
                {
                    "operation_id": "formal_series.rational.divide.compute",
                    "payload": {"left": output["result"], "right": payload},
                },
            )
            assert not quotient.is_error
            assert quotient.structured_content is not None
            claim = SeriesDivideResult.model_validate_json(
                json.dumps(quotient.structured_content["output"])
            )
            assert verify_divide(claim)
            assert [
                value.as_fraction() for value in claim.quotient.coefficients
            ] == list(range(1, 514))
            refused = await client.call_tool(
                "math.run",
                {
                    "operation_id": "formal_series.rational.inverse.compute",
                    "payload": _payload(2049),
                },
            )
            assert refused.is_error
            content = refused.content[0]
            assert isinstance(content, TextContent)
            diagnostic = json.loads(
                content.text.removeprefix("Error executing tool math.run: ")
            )
            assert diagnostic["code"] == "RESOURCE_ADMISSION_REJECTED"
            assert diagnostic["stage"] == "resource_admission"
            recovered = await client.call_tool(
                "math.run",
                {
                    "operation_id": "formal_series.rational.inverse.compute",
                    "payload": _payload(16),
                },
            )
            assert not recovered.is_error

    asyncio.run(scenario())
