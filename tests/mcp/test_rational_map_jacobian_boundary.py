"""Rational map differential values survive the live MCP boundary unchanged."""

import asyncio
import json

from jacobian.math.polynomials.rational_functions.maps._tools import TOOLS
from jacobian.mcp.server import create_server
from mcp import Client


def test_native_mcp_and_derivative_component_composition() -> None:
    async def scenario() -> None:
        tool = TOOLS[0]
        payload = tool.examples[0].input
        request = tool.request_type.model_validate_json(json.dumps(payload))
        async with Client(create_server(), raise_exceptions=False) as client:
            result = await client.call_tool(
                "math.run", {"operation_id": tool.operation_id, "payload": payload}
            )
            assert not result.is_error
            assert result.structured_content is not None
            output = result.structured_content["output"]
            assert output == tool.run(request).model_dump(mode="json")
            composed = await client.call_tool(
                "math.run",
                {
                    "operation_id": "rational_function.gradient.compute",
                    "payload": {"function": output["entries"][0][0]},
                },
            )
            assert not composed.is_error

    asyncio.run(scenario())


def test_sparse_bivariate_power_map_round_trips_into_gradient_consumer() -> None:
    from sympy import symbols

    from jacobian.math.polynomials._conversions import rational_function_from_sympy
    from jacobian.math.polynomials.rational_functions import RationalFunctionMap
    from jacobian.math.polynomials.rational_functions.maps import (
        RationalFunctionMapJacobian,
    )

    x, y = symbols("x y")
    axes = ("x", "y")
    source = RationalFunctionMap(
        source_variables=axes,
        target_coordinates=tuple(f"u{i}" for i in range(27)),
        components=(rational_function_from_sympy(1 / (x + y) ** 33, axes),) * 27,
    )

    async def scenario() -> None:
        async with Client(create_server(), raise_exceptions=False) as client:
            result = await client.call_tool(
                "math.run",
                {
                    "operation_id": TOOLS[0].operation_id,
                    "payload": {"source": source.model_dump(mode="json")},
                },
            )
            assert not result.is_error
            assert result.structured_content is not None
            decoded = RationalFunctionMapJacobian.model_validate_json(
                json.dumps(result.structured_content["output"])
            )
            expected = rational_function_from_sympy(-33 / (x + y) ** 34, axes)
            assert decoded.entries == ((expected, expected),) * 27
            consumed = await client.call_tool(
                "math.run",
                {
                    "operation_id": "rational_function.gradient.compute",
                    "payload": {
                        "function": decoded.entries[0][0].model_dump(mode="json")
                    },
                },
            )
            assert not consumed.is_error

    asyncio.run(scenario())
