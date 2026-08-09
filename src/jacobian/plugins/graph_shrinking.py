"""Untrusted deterministic reducers for finite simple undirected graphs."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from jacobian.contracts.plugin_graphs import GraphShrinkRequest


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
    proposals: list[dict[str, Any]] = []

    if "delete_vertex" in requested:
        for vertex in vertices:
            reduced_edges = tuple(edge for edge in edges if vertex not in edge)
            payload = {
                "graph_schema_version": "1",
                "vertices": [item for item in vertices if item != vertex],
                "edges": [list(edge) for edge in reduced_edges],
            }
            values = {"vertices": len(vertices) - 1, "edges": len(reduced_edges)}
            proposals.append(
                {
                    "reducer": "delete_vertex",
                    "payload": payload,
                    "objectives": {name: values[name] for name in objectives},
                }
            )

    if "delete_edge" in requested:
        for edge in edges:
            payload = {
                "graph_schema_version": "1",
                "vertices": list(vertices),
                "edges": [list(item) for item in edges if item != edge],
            }
            values = {"vertices": len(vertices), "edges": len(edges) - 1}
            proposals.append(
                {
                    "reducer": "delete_edge",
                    "payload": payload,
                    "objectives": {name: values[name] for name in objectives},
                }
            )

    current = {"vertices": len(vertices), "edges": len(edges)}
    return {
        "response_version": "1",
        "current_objectives": {name: current[name] for name in objectives},
        "reductions": proposals,
        "detail": "complete deterministic requested single-deletion neighborhood",
    }
