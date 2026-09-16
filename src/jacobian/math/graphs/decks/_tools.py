"""Vertex-deletion deck operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.graphs.decks._models import (
    VertexDeckRequest,
    VertexDeletionFamily,
)
from jacobian.math.graphs.decks.operations import vertex_deletion_family


def _run_vertex_deleted(request: VertexDeckRequest) -> VertexDeletionFamily:
    return vertex_deletion_family(request.graph)


_PATH_3_EXAMPLE: dict[str, Any] = {
    "graph": {
        "vertices": ["a", "b", "c"],
        "edges": [["a", "b"], ["b", "c"]],
    }
}

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="graph.deck.vertex_deleted.compute",
        title="Compute the complete vertex-deletion family of a graph",
        description=(
            "Delete every vertex of a bounded simple undirected graph once "
            "and return the complete source-bound card family: one card per "
            "source vertex with source-to-card vertex injection and "
            "retained/deleted edge accounting. Before return, replay that "
            "every source edge appears in exactly n-2 cards and every source "
            "vertex in exactly n-1 card domains."
        ),
        request_type=VertexDeckRequest,
        result_type=VertexDeletionFamily,
        run=_run_vertex_deleted,
        tags=("graph", "deck", "vertex-deletion", "reconstruction", "exact"),
        discovery_terms=(
            "vertex deck",
            "vertex-deleted subgraphs",
            "reconstruction",
            "card family",
            "Kelly lemma",
        ),
        examples=(
            OperationExample(
                name="path_p3_deck",
                description="Three-card vertex-deletion family of the path P3.",
                input=_PATH_3_EXAMPLE,
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
