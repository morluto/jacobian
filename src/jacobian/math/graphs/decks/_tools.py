"""Vertex-deletion deck operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.graphs.decks._models import (
    EdgeDeckRequest,
    EdgeDeletionFamily,
    UnlabelledDeck,
    UnlabelledDeckRequest,
    VertexDeckRequest,
    VertexDeletionFamily,
)
from jacobian.math.graphs.decks.operations import (
    edge_deletion_family,
    unlabelled_deck,
    vertex_deletion_family,
)


def _run_vertex_deleted(request: VertexDeckRequest) -> VertexDeletionFamily:
    return vertex_deletion_family(request.graph)


_PATH_3_EXAMPLE: dict[str, Any] = {
    "graph": {
        "vertices": ["a", "b", "c"],
        "edges": [["a", "b"], ["b", "c"]],
    }
}


def _run_edge_deleted(request: EdgeDeckRequest) -> EdgeDeletionFamily:
    return edge_deletion_family(request.graph)


def _run_unlabelled(request: UnlabelledDeckRequest) -> UnlabelledDeck:
    return unlabelled_deck(request.deck)


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="graph.deck.vertex_deleted.compute",
        title="Compute the complete vertex-deletion family of a graph",
        description=(
            "Delete every vertex of a simple undirected graph once (at most 64 "
            "vertices and 130000 aggregate retained card edges) and return the "
            "complete source-bound card family: one card per "
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
    MathTool(
        operation_id="graph.deck.edge_deleted.compute",
        title="Compute the complete edge-deletion family of a graph",
        description=(
            "Delete each source edge exactly once and return source-bound cards "
            "retaining every isolated vertex and the deleted-edge key; aggregate "
            "retained card edges are bounded by 130000."
        ),
        request_type=EdgeDeckRequest,
        result_type=EdgeDeletionFamily,
        run=_run_edge_deleted,
        tags=("graph", "deck", "edge-deletion", "exact"),
        discovery_terms=("edge deck", "edge-deleted cards", "graph deck"),
        examples=(
            OperationExample(
                name="triangle_edge_deck",
                description="Compute the three edge-deleted cards of a triangle; every card retains all three source vertices.",
                input={
                    "graph": {
                        "vertices": ["a", "b", "c"],
                        "edges": [["a", "b"], ["a", "c"], ["b", "c"]],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="graph.deck.unlabelled.compute",
        title="Compute the unlabelled multiset quotient of an edge deck",
        description=(
            "Group source edge-deletion cards by exact graph isomorphism and retain "
            "each representative with its positive multiplicity and source card "
            "indices; quotient work admits at most 10 source vertices and "
            "2000000 comparison units."
        ),
        request_type=UnlabelledDeckRequest,
        result_type=UnlabelledDeck,
        run=_run_unlabelled,
        tags=("graph", "deck", "isomorphism", "multiset", "exact"),
        discovery_terms=("unlabelled deck", "deck quotient", "deck multiplicities"),
        examples=(
            OperationExample(
                name="path_edge_quotient",
                description="Quotient the two edge-deleted cards of P3; the cards are isomorphic and therefore have multiplicity two.",
                input={
                    "deck": {
                        "source": {
                            "vertices": ["a", "b", "c"],
                            "edges": [["a", "b"], ["b", "c"]],
                        },
                        "cards": [
                            {
                                "deleted_edge": ["a", "b"],
                                "card": {
                                    "vertices": ["a", "b", "c"],
                                    "edges": [["b", "c"]],
                                },
                                "retained_vertices": ["a", "b", "c"],
                                "retained_edge_count": 1,
                            },
                            {
                                "deleted_edge": ["b", "c"],
                                "card": {
                                    "vertices": ["a", "b", "c"],
                                    "edges": [["a", "b"]],
                                },
                                "retained_vertices": ["a", "b", "c"],
                                "retained_edge_count": 1,
                            },
                        ],
                    }
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
