"""Native/MCP schema parity, recovery and canonical scalar-tensor reuse."""

import asyncio
import json

from jsonschema import validate

from jacobian.math.geometry.differential.metrics._tools import TOOLS
from jacobian.mcp.server import create_server
from mcp import Client


def test_live_rational_curvature_and_tensor_composition() -> None:
    async def scenario() -> None:
        tool = TOOLS[0]
        payload = tool.examples[0].input
        validate(payload, tool.request_type.model_json_schema())
        request = tool.request_type.model_validate_json(json.dumps(payload))
        async with Client(create_server(), raise_exceptions=False) as client:
            invalid = json.loads(json.dumps(payload))
            invalid["metric"]["tensor"]["components"][3]["numerator"] = {"terms": []}
            failure = await client.call_tool(
                "math.run", {"operation_id": tool.operation_id, "payload": invalid}
            )
            assert failure.is_error
            response = await client.call_tool(
                "math.run", {"operation_id": tool.operation_id, "payload": payload}
            )
            assert not response.is_error
            assert response.structured_content is not None
            output = response.structured_content["output"]
            validate(output, tool.result_type.model_json_schema())
            assert output == tool.run(request).model_dump(mode="json")
            again = await client.call_tool(
                "math.run",
                {
                    "operation_id": tool.operation_id,
                    "payload": {"metric": output["metric"]},
                },
            )
            assert not again.is_error
            assert again.structured_content is not None
            assert again.structured_content["output"] == output
            assert output["scalar_curvature"]["variance"] == []
            assert output["scalar_curvature"]["retained_nonzero_denominators"]
            differentiated = await client.call_tool(
                "math.run",
                {
                    "operation_id": "differential_geometry.rational_tensor.lie_derivative.compute",
                    "payload": {
                        "vector_field": {
                            "coordinate_axis": ["r", "theta"],
                            "variance": ["CONTRAVARIANT"],
                            "components": output["metric"]["tensor"]["components"][:2],
                        },
                        "tensor": output["scalar_curvature"],
                    },
                },
            )
            assert not differentiated.is_error
            assert differentiated.structured_content is not None
            assert (
                differentiated.structured_content["output"]["lie_derivative"]
                == output["scalar_curvature"]
            )

    asyncio.run(scenario())
