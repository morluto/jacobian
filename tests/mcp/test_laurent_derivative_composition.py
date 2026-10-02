"""Repeated public Laurent differentiation consumes actual serialized output."""

import asyncio
import json
from typing import Any

import pytest
import sympy
from mcp.types import TextContent

from jacobian.math.polynomials._conversions import rational_function_from_sympy
from jacobian.math.polynomials.values import RationalFunction
from jacobian.mcp.server import create_server
from mcp import Client


@pytest.mark.parametrize(
    "operation_id",
    ["rational_function.gradient.compute", "rational_function_map.jacobian.compute"],
)
def test_laurent_public_derivative_chain(operation_id: str) -> None:
    async def scenario() -> None:
        def payload(function: dict[str, Any]) -> dict[str, Any]:
            if operation_id == "rational_function.gradient.compute":
                return {"function": function}
            return {
                "source": {
                    "source_variables": ["x"],
                    "target_coordinates": ["u"],
                    "components": [function],
                }
            }

        x = sympy.Symbol("x")
        current = rational_function_from_sympy(x**-63, ("x",)).model_dump(mode="json")
        async with Client(create_server(), raise_exceptions=False) as client:
            for expression in (-63 * x**-64, 4032 * x**-65):
                response = await client.call_tool(
                    "math.run",
                    {"operation_id": operation_id, "payload": payload(current)},
                )
                assert not response.is_error
                assert response.structured_content is not None
                output = response.structured_content["output"]
                current = (
                    output["partial_derivatives"][0]
                    if "partial_derivatives" in output
                    else output["entries"][0][0]
                )
                restored = RationalFunction.model_validate_json(json.dumps(current))
                assert restored == rational_function_from_sympy(expression, ("x",))
            boundary = rational_function_from_sympy(x**-127, ("x",)).model_dump(
                mode="json"
            )
            accepted = await client.call_tool(
                "math.run", {"operation_id": operation_id, "payload": payload(boundary)}
            )
            assert not accepted.is_error
            assert accepted.structured_content is not None
            output = accepted.structured_content["output"]
            current = (
                output["partial_derivatives"][0]
                if "partial_derivatives" in output
                else output["entries"][0][0]
            )
            assert RationalFunction.model_validate_json(
                json.dumps(current)
            ) == rational_function_from_sympy(-127 * x**-128, ("x",))
            refused = await client.call_tool(
                "math.run", {"operation_id": operation_id, "payload": payload(current)}
            )
            assert refused.is_error
            assert isinstance(refused.content[0], TextContent)
            text = refused.content[0].text
            diagnostic = json.loads(text[text.index("{") :])
            assert diagnostic["code"] == "RESOURCE_ADMISSION_REJECTED"
            assert (
                diagnostic["errors"][0]["code"]
                == operation_id.removesuffix(".compute") + ".result_exponent"
            )

    asyncio.run(scenario())
