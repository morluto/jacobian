"""Public wide-source recognition and unchanged derivative consumption."""

import asyncio
import json
from typing import Any

import sympy
from mcp.types import TextContent

from jacobian.math.polynomials._conversions import rational_function_from_sympy
from jacobian.math.polynomials.rational_functions.gradient import (
    RationalFunctionGradient,
)
from jacobian.mcp.server import create_server
from mcp import Client


def test_rational_source_envelope_through_live_mcp() -> None:
    async def scenario() -> None:
        x = sympy.Symbol("x")
        source = rational_function_from_sympy(x**65, ("unused", "x")).model_dump(
            mode="json"
        )
        source_map: dict[str, Any] = {
            "source_variables": ["unused", "x"],
            "target_coordinates": ["u"],
            "components": [source],
        }
        async with Client(create_server(), raise_exceptions=False) as client:
            response = await client.call_tool(
                "math.run",
                {
                    "operation_id": "rational_function.gradient.compute",
                    "payload": {"function": source},
                },
            )
            assert not response.is_error
            assert response.structured_content is not None
            output = response.structured_content["output"]
            restored = RationalFunctionGradient.model_validate_json(json.dumps(output))
            assert restored.partial_derivatives[1] == rational_function_from_sympy(
                65 * x**64, ("unused", "x")
            )
            repeated = await client.call_tool(
                "math.run",
                {
                    "operation_id": "rational_function.gradient.compute",
                    "payload": {"function": output["partial_derivatives"][1]},
                },
            )
            assert not repeated.is_error
            assert repeated.structured_content is not None
            assert repeated.structured_content["output"]["partial_derivatives"][
                1
            ] == rational_function_from_sympy(4160 * x**63, ("unused", "x")).model_dump(
                mode="json"
            )
            jacobian = await client.call_tool(
                "math.run",
                {
                    "operation_id": "rational_function_map.jacobian.compute",
                    "payload": {"source": source_map},
                },
            )
            assert not jacobian.is_error
            assert jacobian.structured_content is not None
            assert jacobian.structured_content["output"]["entries"] == [
                output["partial_derivatives"]
            ]
            source_map["components"] = [
                rational_function_from_sympy(x**128, ("unused", "x")).model_dump(
                    mode="json"
                )
            ]
            composition = await client.call_tool(
                "math.run",
                {
                    "operation_id": "rational_function_map.compose.compute",
                    "payload": {
                        "inner": source_map,
                        "outer": {
                            "source_variables": ["u"],
                            "target_coordinates": ["v"],
                            "components": [
                                rational_function_from_sympy(1, ("u",)).model_dump(
                                    mode="json"
                                )
                            ],
                        },
                    },
                },
            )
            assert not composition.is_error
            assert composition.structured_content is not None
            assert composition.structured_content["output"]["composite"][
                "components"
            ] == [
                rational_function_from_sympy(1, ("unused", "x")).model_dump(mode="json")
            ]
            source_map["components"] = [
                rational_function_from_sympy(x**-128, ("unused", "x")).model_dump(
                    mode="json"
                )
            ]
            guard_refusal = await client.call_tool(
                "math.run",
                {
                    "operation_id": "rational_function_map.compose.compute",
                    "payload": {
                        "inner": source_map,
                        "outer": {
                            "source_variables": ["u"],
                            "target_coordinates": ["v"],
                            "components": [
                                rational_function_from_sympy(1, ("u",)).model_dump(
                                    mode="json"
                                )
                            ],
                        },
                    },
                },
            )
            assert guard_refusal.is_error
            assert isinstance(guard_refusal.content[0], TextContent)
            guard_text = guard_refusal.content[0].text
            guard_diagnostic = json.loads(guard_text[guard_text.index("{") :])
            assert guard_diagnostic["code"] == "RESOURCE_ADMISSION_REJECTED"
            assert (
                guard_diagnostic["errors"][0]["code"]
                == "rational_function_map.compose.guard_exponent"
            )
            unreduced = dict(source)
            unreduced["denominator"] = source["numerator"]
            refused = await client.call_tool(
                "math.run",
                {
                    "operation_id": "rational_function.gradient.compute",
                    "payload": {"function": unreduced},
                },
            )
            assert refused.is_error
            assert isinstance(refused.content[0], TextContent)
            text = refused.content[0].text
            diagnostic = json.loads(text[text.index("{") :])
            assert diagnostic["code"] == "INVALID_REQUEST"
            assert diagnostic["errors"][0]["code"] == "polynomial.not_coprime"

    asyncio.run(scenario())
