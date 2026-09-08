import asyncio
import json

from tests.mcp.test_direct_operations import _server

from jacobian.math.combinatorics.designs.incidence_structures.operations import (
    containment_profile,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs import (
    FiniteHypergraph,
    parameters,
)
from mcp import Client


def test_hypergraph_containment_mcp_roundtrip() -> None:
    async def scenario() -> None:
        operation = "incidence.containment_profiles.compute"
        async with Client(_server(operation), raise_exceptions=True) as client:
            vertices = [f"v{i}" for i in range(256)]
            edges = [[f"e{i}", [vertices[i % 256]]] for i in range(12000)]
            payload = {"incidence": {"vertices": vertices, "edges": edges}, "t": 1}
            response = await client.call_tool(operation, payload)
            assert response.structured_content is not None
            result = response.structured_content
            assert result["total_multiplicity"] == 12000
            source = FiniteHypergraph.model_validate_json(
                json.dumps(payload["incidence"])
            )
            native = containment_profile(source, 1)
            restored = type(native).model_validate_json(json.dumps(result))
            assert restored == native
            assert isinstance(restored.incidence, FiniteHypergraph)
            assert parameters(restored.incidence).vertex_count == 256

    asyncio.run(scenario())
