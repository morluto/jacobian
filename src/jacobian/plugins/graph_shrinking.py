"""Untrusted deterministic reducers for finite simple undirected graphs."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from jacobian.contracts.graph_isomorphism import SimpleUndirectedGraph
from jacobian.contracts.plugin_graphs import GraphShrinkRequest
from jacobian.contracts.shrinking import PluginReductionResponse, ReductionProposal


def reduce_simple_graph(request: dict[str, Any]) -> dict[str, Any]:
    """Propose every requested single deletion in canonical order."""

    try:
        selected = GraphShrinkRequest.model_validate(request)
    except ValidationError as exc:
        raise ValueError("graph shrinking request does not match its contract") from exc

    target = selected.target
    vertices = target.vertices
    edges = target.edges
    requested = selected.reducers
    objectives = selected.objectives
    proposals: list[ReductionProposal] = []

    if "delete_vertex" in requested:
        for vertex in vertices:
            reduced_edges = tuple(edge for edge in edges if vertex not in edge)
            payload = SimpleUndirectedGraph(
                vertices=tuple(item for item in vertices if item != vertex),
                edges=reduced_edges,
            )
            values = {"vertices": len(vertices) - 1, "edges": len(reduced_edges)}
            proposals.append(
                ReductionProposal(
                    reducer="delete_vertex",
                    payload=payload.model_dump(mode="json"),
                    objectives={name: values[name] for name in objectives},
                )
            )

    if "delete_edge" in requested:
        for edge in edges:
            payload = SimpleUndirectedGraph(
                vertices=vertices,
                edges=tuple(item for item in edges if item != edge),
            )
            values = {"vertices": len(vertices), "edges": len(edges) - 1}
            proposals.append(
                ReductionProposal(
                    reducer="delete_edge",
                    payload=payload.model_dump(mode="json"),
                    objectives={name: values[name] for name in objectives},
                )
            )

    current = {"vertices": len(vertices), "edges": len(edges)}
    return PluginReductionResponse(
        current_objectives={name: current[name] for name in objectives},
        reductions=tuple(proposals),
        detail="complete deterministic requested single-deletion neighborhood",
    ).model_dump(mode="json")
