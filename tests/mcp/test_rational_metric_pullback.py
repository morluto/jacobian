"""Live MCP parity and tensor composition for rational metric pullback."""

import asyncio
import json

from jsonschema import validate
from sympy import symbols

from jacobian.math.geometry.differential.metrics import RationalCoordinateMetric
from jacobian.math.geometry.differential.pullback._models import (
    RationalMetricPullbackRequest,
)
from jacobian.math.geometry.differential.pullback._tools import TOOLS
from jacobian.math.geometry.differential.values import RationalCoordinateTensor
from jacobian.math.polynomials._conversions import rational_function_from_sympy
from jacobian.math.polynomials.rational_functions.values import RationalFunctionMap
from jacobian.mcp.server import create_server
from mcp import Client


def test_live_pullback_matches_native_and_composes_as_tensor() -> None:
    async def scenario() -> None:
        x, u = symbols("x u")
        axis = ("x",)
        target = ("u",)
        q = rational_function_from_sympy
        metric = RationalCoordinateMetric(
            tensor=RationalCoordinateTensor(
                coordinate_axis=target,
                variance=("COVARIANT", "COVARIANT"),
                components=(q(u**2 + 1, target),),
            )
        )
        mapping = RationalFunctionMap(
            source_variables=axis,
            target_coordinates=target,
            components=(q(x, axis),),
        )
        request = RationalMetricPullbackRequest(metric=metric, map=mapping)
        tool = TOOLS[0]
        payload = request.model_dump(mode="json")
        validate(payload, tool.request_type.model_json_schema())
        expected = tool.run(request).model_dump(mode="json")
        async with Client(create_server(), raise_exceptions=False) as client:
            response = await client.call_tool(
                "math.run", {"operation_id": tool.operation_id, "payload": payload}
            )
            assert not response.is_error
            assert response.structured_content is not None
            output = response.structured_content["output"]
            assert output == expected
            decoded = RationalMetricPullbackRequest.model_validate_json(
                json.dumps(payload)
            )
            assert decoded == request
            # Feed the returned tensor into the existing tensor operation so the
            # MCP result is exercised as a typed producer-consumer value.
            consumer = await client.call_tool(
                "math.run",
                {
                    "operation_id": "differential_geometry.rational_tensor.lie_derivative.compute",
                    "payload": {
                        "vector_field": {
                            "coordinate_axis": ["x"],
                            "variance": ["CONTRAVARIANT"],
                            "components": [
                                {
                                    "variables": ["x"],
                                    "numerator": {
                                        "terms": [
                                            {
                                                "coefficient": {"num": "1", "den": "1"},
                                                "exponents": [0],
                                            }
                                        ]
                                    },
                                    "denominator": {
                                        "terms": [
                                            {
                                                "coefficient": {"num": "1", "den": "1"},
                                                "exponents": [0],
                                            }
                                        ]
                                    },
                                }
                            ],
                        },
                        "tensor": output["pullback"],
                    },
                },
            )
            assert not consumer.is_error

    asyncio.run(scenario())
