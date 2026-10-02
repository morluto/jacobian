"""Specialized resultant resource refusals stay distinct and recoverable."""

import asyncio
import json

from jsonschema import validate
from mcp.types import TextContent

from jacobian.math.polynomials.multivariate import verify_multivariate_resultant
from jacobian.math.polynomials.multivariate._resultant import (
    MultivariateResultantResult,
)
from jacobian.mcp.server import create_server
from mcp import Client


def _payload(degree: int, variable: str = "x") -> dict[str, object]:
    source = {
        "variables": ["x", "y"],
        "polynomial": {
            "terms": [
                {"coefficient": {"num": "1", "den": "1"}, "exponents": [degree, 0]}
            ]
        },
    }
    return {"left": source, "right": source, "elimination_variable": variable}


def test_resultant_invalid_resource_and_success_outcomes_in_one_session() -> None:
    async def scenario() -> None:
        operation_id = "polynomial.multivariate.resultant.compute"
        async with Client(create_server(), raise_exceptions=False) as client:
            for payload, expected, code in (
                (
                    _payload(32, "z"),
                    "INVALID_REQUEST",
                    "polynomial.multivariate_contract",
                ),
                (
                    _payload(33),
                    "RESOURCE_ADMISSION_REJECTED",
                    "polynomial.multivariate_resultant.degree_budget",
                ),
            ):
                failed = await client.call_tool(
                    "math.run", {"operation_id": operation_id, "payload": payload}
                )
                assert failed.is_error
                assert isinstance(failed.content[0], TextContent)
                text = failed.content[0].text
                diagnostic = json.loads(text[text.index("{") :])
                assert diagnostic["code"] == expected
                assert diagnostic["errors"][0]["code"] == code
                if expected == "RESOURCE_ADMISSION_REJECTED":
                    assert diagnostic["stage"] == "resource_admission"
                    assert "66" in diagnostic["errors"][0]["message"]
                    assert "64" in diagnostic["errors"][0]["message"]
                    assert diagnostic["errors"][0]["location"] == [
                        "elimination_variable"
                    ]
            success = await client.call_tool(
                "math.run", {"operation_id": operation_id, "payload": _payload(32)}
            )
            assert not success.is_error
            assert success.structured_content is not None
            output = success.structured_content["output"]
            validate(
                output,
                MultivariateResultantResult.model_json_schema(mode="serialization"),
            )
            decoded = MultivariateResultantResult.model_validate_json(
                json.dumps(output), strict=True
            )
            assert decoded.resultant.kind == "POLYNOMIAL"
            assert decoded.resultant.value.variables == ("y",)
            assert not decoded.resultant.value.polynomial.terms
            assert verify_multivariate_resultant(decoded)

    asyncio.run(scenario())
