import asyncio
import json

import pytest
from tests.mcp.test_direct_operations import _server

from jacobian.math.combinatorics.designs.incidence_structures.operations import (
    containment_profile,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs import (
    FiniteHypergraph,
    parameters,
)
from mcp import Client


@pytest.mark.parametrize(
    ("point_count", "edge_count", "block_size", "order", "expected_total"),
    [(256, 12_000, 1, 1, 12_000), (14, 2_500, 14, 7, 8_580_000)],
)
def test_hypergraph_containment_mcp_roundtrip(
    point_count: int, edge_count: int, block_size: int, order: int, expected_total: int
) -> None:
    async def scenario() -> None:
        operation = "incidence.containment_profiles.compute"
        async with Client(_server(operation), raise_exceptions=True) as client:
            vertices = [f"v{i}" for i in range(point_count)]
            edges = [
                [f"e{i}", [vertices[(i + j) % point_count] for j in range(block_size)]]
                for i in range(edge_count)
            ]
            payload = {"incidence": {"vertices": vertices, "edges": edges}, "t": order}
            response = await client.call_tool(operation, payload)
            assert response.structured_content is not None
            result = response.structured_content
            assert result["total_multiplicity"] == expected_total
            source = FiniteHypergraph.model_validate_json(
                json.dumps(payload["incidence"])
            )
            native = containment_profile(source, order)
            restored = type(native).model_validate_json(json.dumps(result))
            assert restored == native
            assert isinstance(restored.incidence, FiniteHypergraph)
            assert parameters(restored.incidence).vertex_count == point_count

    asyncio.run(scenario())
