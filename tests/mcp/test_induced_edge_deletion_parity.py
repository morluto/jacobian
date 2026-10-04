"""Native and MCP parity for the induced edge-deletion profile."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence

from jacobian.math.graphs.coloring.induced_edge_deletion_profile._models import (
    InducedEdgeDeletionProfileRequest,
    InducedEdgeDeletionProfileResult,
)
from jacobian.math.graphs.coloring.induced_edge_deletion_profile.operations import (
    compute_induced_edge_deletion_profile,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _graph(
    vertices: Sequence[str], edges: Sequence[Sequence[str]]
) -> SimpleUndirectedGraph:
    return SimpleUndirectedGraph(
        vertices=tuple(vertices),
        edges=tuple((a, b) if a < b else (b, a) for a, b in edges),
    )


def test_native_mcp_parity() -> None:
    """The MCP projection must return exactly what the kernel computes.

    This exercises the real client, ``math.run`` request, and response decoding;
    a transport or projection regression would fail here.
    """
    g = _graph(["a", "b", "c"], [("a", "b"), ("a", "c"), ("b", "c")])
    native = compute_induced_edge_deletion_profile(g, 2)
    request = InducedEdgeDeletionProfileRequest(graph=g, r=2)

    async def scenario() -> object:
        from jacobian.mcp.server import create_server
        from mcp import Client

        async with Client(create_server(), raise_exceptions=True) as client:
            result = await client.call_tool(
                "math.run",
                {
                    "operation_id": (
                        "graph.coloring.induced_edge_deletion_profile.compute"
                    ),
                    "payload": request.model_dump(mode="json"),
                },
            )
        assert result.is_error is False
        assert isinstance(result.structured_content, dict)
        return result.structured_content["output"]

    output = json.loads(json.dumps(asyncio.run(scenario())))
    assert InducedEdgeDeletionProfileResult.model_validate(output) == native
