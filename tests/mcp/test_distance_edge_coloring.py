"""Native distance colouring still composes; the projection is not catalogued."""

import asyncio
import json

from jsonschema import validate

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.math.geometry.exact.distance_edge_coloring import (
    compute_distance_edge_coloring,
)
from jacobian.math.geometry.exact.distance_edge_coloring._models import (
    DistanceEdgeColoringRequest,
    DistanceEdgeColoringResult,
)
from jacobian.mcp.server import create_server
from mcp import Client

_UNIT_SQUARE = {
    "configuration": {
        "points": [
            {
                "label": label,
                "coordinates": [
                    {"num": str(x), "den": "1"},
                    {"num": str(y), "den": "1"},
                ],
            }
            for label, x, y in (
                ("a", 0, 0),
                ("b", 1, 0),
                ("c", 0, 1),
                ("d", 1, 1),
            )
        ]
    }
}


def test_distance_coloring_stays_native_and_composes_with_parameters() -> None:
    public_ids = {tool.operation_id for tool in BUILTIN_TOOLS}
    assert "geometry.points.distance_edge_coloring.compute" not in public_ids
    validate(_UNIT_SQUARE, DistanceEdgeColoringRequest.model_json_schema())
    request = DistanceEdgeColoringRequest.model_validate_json(json.dumps(_UNIT_SQUARE))
    native = compute_distance_edge_coloring(request.configuration).model_dump(
        mode="json"
    )
    validate(native, DistanceEdgeColoringResult.model_json_schema())

    async def scenario() -> None:
        async with Client(create_server(), raise_exceptions=False) as client:
            parameters = await client.call_tool(
                "math.run",
                {
                    "operation_id": "hypergraph.parameters.compute",
                    "payload": {"hypergraph": native["coloring"]["hypergraph"]},
                },
            )
            assert not parameters.is_error
            assert parameters.structured_content is not None
            assert parameters.structured_content["output"]["edge_count"] == 6

    asyncio.run(scenario())
