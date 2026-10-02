"""Large exact polynomial invariants through native, schema and live MCP seams."""

import asyncio
import json
from typing import Any

from jsonschema import validate
from mcp.types import TextContent

from jacobian.catalog.models import MathTool
from jacobian.math.polynomials._invariants import POLYNOMIAL_INVARIANT_OPERATIONS
from jacobian.math.polynomials._models import (
    PolynomialDiscriminantResult,
    PolynomialResultantResult,
)
from jacobian.math.polynomials.operations import (
    verify_polynomial_discriminant,
    verify_polynomial_resultant,
)
from jacobian.mcp.server import create_server
from mcp import Client


def _polynomial(exponents: list[list[int]]) -> dict[str, object]:
    return {
        "domain": "QQ",
        "variables": ["x", "y", "z", "w"],
        "polynomial": {
            "terms": [
                {"exponents": e, "coefficient": {"num": "1", "den": "1"}}
                for e in exponents
            ]
        },
    }


def test_large_invariants_mcp_schema_parity_and_recovery() -> None:
    async def scenario() -> None:
        tools: dict[str, MathTool[Any, Any]] = {
            tool.operation_id: tool for tool in POLYNOMIAL_INVARIANT_OPERATIONS
        }
        tail = [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        left = _polynomial([[1, 0, 0, 0], *tail])
        requests = (
            (
                "polynomial.compute.resultant",
                {
                    "left": left,
                    "right": _polynomial([[44, 0, 0, 0]]),
                    "elimination_variable": "x",
                },
            ),
            (
                "polynomial.compute.discriminant",
                {"polynomial": _polynomial([[45, 0, 0, 0], *tail]), "variable": "x"},
            ),
        )
        async with Client(create_server(), raise_exceptions=False) as client:
            refused = await client.call_tool(
                "math.run",
                {
                    "operation_id": "polynomial.compute.resultant",
                    "payload": {
                        "left": left,
                        "right": _polynomial([[90, 0, 0, 0]]),
                        "elimination_variable": "x",
                    },
                },
            )
            assert refused.is_error
            assert isinstance(refused.content[0], TextContent)
            text = refused.content[0].text
            error = json.loads(text[text.index("{") :])
            assert error["code"] == "RESOURCE_ADMISSION_REJECTED"
            assert error["errors"][0]["code"] == "polynomial.invariant_budget"
            for operation_id, payload in requests:
                tool = tools[operation_id]
                validate(payload, tool.request_type.model_json_schema())
                native = tool.run(
                    tool.request_type.model_validate_json(json.dumps(payload))
                )
                reply = await client.call_tool(
                    "math.run", {"operation_id": operation_id, "payload": payload}
                )
                assert not reply.is_error
                assert reply.structured_content is not None
                output = reply.structured_content["output"]
                assert output == native.model_dump(mode="json")
                validate(output, tool.result_type.model_json_schema())
                decoded = tool.result_type.model_validate_json(json.dumps(output))
                if isinstance(decoded, PolynomialResultantResult):
                    assert decoded.resultant.kind == "POLYNOMIAL"
                    assert len(decoded.resultant.value.polynomial.terms) == 1035
                    assert verify_polynomial_resultant(decoded)
                else:
                    assert isinstance(decoded, PolynomialDiscriminantResult)
                    assert decoded.discriminant.kind == "POLYNOMIAL"
                    assert len(decoded.discriminant.value.polynomial.terms) == 1035
                    assert verify_polynomial_discriminant(decoded)

    asyncio.run(scenario())
