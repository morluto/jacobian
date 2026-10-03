"""Native and MCP parity for the induced edge-deletion profile."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from itertools import combinations, product

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


def _brute_is_r_colorable(
    vertices: list[str], edges: list[tuple[str, str]], r: int
) -> bool:
    if not edges:
        return True
    if r >= len(vertices):
        return True
    if r == 1:
        return False
    idx = {v: i for i, v in enumerate(vertices)}
    for coloring in product(range(r), repeat=len(vertices)):
        if all(coloring[idx[a]] != coloring[idx[b]] for a, b in edges):
            return True
    return False


def _brute_min_deletions(
    vertices: list[str], edges: list[tuple[str, str]], r: int
) -> tuple[int, tuple[tuple[str, str], ...]]:
    sorted_edges = sorted(edges)
    m = len(sorted_edges)
    for k in range(m + 1):
        for combo in combinations(range(m), k):
            remaining = [e for i, e in enumerate(sorted_edges) if i not in combo]
            if _brute_is_r_colorable(vertices, remaining, r):
                deleted = tuple(sorted_edges[i] for i in combo)
                return k, deleted
    return m, tuple(sorted_edges)


def _exhaustive_profile_brute(
    graph: SimpleUndirectedGraph, r: int
) -> list[tuple[tuple[str, ...], int, tuple[tuple[str, str], ...]]]:
    sorted_vertices = sorted(graph.vertices)
    sorted_edges = tuple(sorted(graph.edges))
    rows: list[tuple[tuple[str, ...], int, tuple[tuple[str, str], ...]]] = []
    n = len(sorted_vertices)
    for size in range(n + 1):
        for subset in combinations(sorted_vertices, size):
            subset_set = set(subset)
            induced = [
                e for e in sorted_edges if e[0] in subset_set and e[1] in subset_set
            ]
            k, deleted = _brute_min_deletions(list(subset), induced, r)
            rows.append((tuple(subset), k, deleted))
    return rows


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
