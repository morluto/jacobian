"""Vertex-deletion deck operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.graphs.decks._models import (
    MAX_KELLY_DECK_TOTAL_WORK,
    AnonymousCardDegreeProfile,
    AnonymousCardDegreeProfileRequest,
    AnonymousGraphCardMultiset,
    AnonymousGraphCardMultisetRequest,
    EdgeDeckRequest,
    EdgeDeletionFamily,
    UnlabelledDeck,
    UnlabelledDeckRequest,
    UnlabelledVertexDeck,
    UnlabelledVertexDeckRequest,
    VertexDeckDegreeMultisetRequest,
    VertexDeckEdgeCount,
    VertexDeckEdgeCountRequest,
    VertexDeckInducedSubgraphCount,
    VertexDeckInducedSubgraphCountRequest,
    VertexDeckRequest,
    VertexDeckSubgraphCount,
    VertexDeckSubgraphCountRequest,
    VertexDeletionFamily,
)
from jacobian.math.graphs.decks.operations import (
    anonymous_card_degree_profile,
    anonymous_graph_card_multiset,
    edge_deletion_family,
    unlabelled_deck,
    unlabelled_vertex_deck,
    vertex_deck_degree_multiset,
    vertex_deck_edge_count,
    vertex_deck_induced_subgraph_count,
    vertex_deck_subgraph_count,
    vertex_deletion_family,
)
from jacobian.math.graphs.realization._models import DegreeSequence


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


def _run_unlabelled_vertex(
    request: UnlabelledVertexDeckRequest,
) -> UnlabelledVertexDeck:
    return unlabelled_vertex_deck(request.deck)


def _run_anonymous_card_degree_profile(
    request: AnonymousCardDegreeProfileRequest,
) -> AnonymousCardDegreeProfile:
    # The request's before-validator admits the combined envelope before nested
    # card parsing; that parser then establishes canonical form exactly once.
    return anonymous_card_degree_profile(request.multiset)


def _run_vertex_deck_induced_pattern_count(
    request: VertexDeckInducedSubgraphCountRequest,
) -> VertexDeckInducedSubgraphCount:
    return vertex_deck_induced_subgraph_count(request.deck, request.pattern)


def _run_vertex_deck_subgraph_count(
    request: VertexDeckSubgraphCountRequest,
) -> VertexDeckSubgraphCount:
    return vertex_deck_subgraph_count(request.deck, request.pattern)


def _run_vertex_deck_edge_count(
    request: VertexDeckEdgeCountRequest,
) -> VertexDeckEdgeCount:
    return vertex_deck_edge_count(request.deck)


def _run_vertex_deck_degree_multiset(
    request: VertexDeckDegreeMultisetRequest,
) -> DegreeSequence:
    return vertex_deck_degree_multiset(request.deck)


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="graph.deck.card_invariant_profile.compute",
        title="Profile anonymous graph cards by degree multiset",
        description=(
            "Group the degree multisets of canonical anonymous graph-card "
            "representatives, summing each class's exact positive multiplicity. "
            "Retain the declared card order, including for an empty input. This "
            "is an invariant profile of the supplied cards; it does not assert "
            "that they form a realizable graph deck or identify a source."
        ),
        request_type=AnonymousCardDegreeProfileRequest,
        result_type=AnonymousCardDegreeProfile,
        run=_run_anonymous_card_degree_profile,
        tags=("graph", "deck", "anonymous", "degree-multiset", "invariant", "exact"),
        discovery_terms=(
            "anonymous graph card degree profile",
            "degree multiset histogram of cards",
            "degree invariant across card classes",
        ),
        examples=(
            OperationExample(
                name="empty_and_edge_card_profile",
                description=(
                    "Count the degree multisets of an empty three-vertex card and "
                    "two copies of a one-edge card; all representatives must be "
                    "canonical cards of the declared order three."
                ),
                input={
                    "multiset": {
                        "card_order": 3,
                        "classes": [
                            {
                                "representative": {
                                    "vertices": ["v00", "v01", "v02"],
                                    "edges": [],
                                },
                                "multiplicity": "1",
                            },
                            {
                                "representative": {
                                    "vertices": ["v00", "v01", "v02"],
                                    "edges": [["v01", "v02"]],
                                },
                                "multiplicity": "2",
                            },
                        ],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="graph.deck.from_cards.construct",
        title="Canonicalize an anonymous multiset of graph cards",
        description=(
            "Take an explicitly ordered multiset of simple graph cards, quotient "
            "by exact isomorphism, and return canonical representatives and exact "
            "positive multiplicities. This value retains no source graph or "
            "deletion identifiers and makes no claim that the multiset is "
            "realizable as a graph deck. The empty multiset retains its declared "
            "card_order. Exact vertex-permutation work and output are admitted "
            "before canonicalization."
        ),
        request_type=AnonymousGraphCardMultisetRequest,
        result_type=AnonymousGraphCardMultiset,
        run=lambda request: anonymous_graph_card_multiset(request),
        tags=("graph", "deck", "anonymous", "multiset", "isomorphism", "exact"),
        discovery_terms=(
            "anonymous graph card multiset",
            "unlabelled graph cards",
            "deck realizability input",
        ),
        examples=(
            OperationExample(
                name="anonymous_two_vertex_cards",
                description=(
                    "Canonicalize two relabelings of the same one-edge graph into "
                    "one class of multiplicity two; each card must have the "
                    "declared order two."
                ),
                input={
                    "card_order": 2,
                    "cards": [
                        {"vertices": ["a", "b"], "edges": [["a", "b"]]},
                        {"vertices": ["x", "y"], "edges": [["x", "y"]]},
                    ],
                },
            ),
        ),
    ),
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
            "Group source edge-deletion cards by exact graph isomorphism, "
            "canonicalizing each card to the least adjacency encoding across all "
            "vertex permutations, and retain each representative with its "
            "positive multiplicity and source card indices; deleted source edges "
            "are recovered from the card indices on the source edge axis. Quotient "
            "work admits at most 10 source vertices and 2000000 exact "
            "permutation-canonicalization work units."
        ),
        request_type=UnlabelledDeckRequest,
        result_type=UnlabelledDeck,
        run=_run_unlabelled,
        tags=(
            "graph",
            "deck",
            "edge-deletion",
            "isomorphism",
            "multiset",
            "exact",
        ),
        discovery_terms=(
            "unlabelled deck",
            "unlabelled edge deck",
            "deck quotient",
            "deck multiplicities",
            "edge deck multiplicities",
        ),
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
    MathTool(
        operation_id="graph.deck.vertex.unlabelled.compute",
        title="Compute the unlabelled multiset quotient of a vertex deck",
        description=(
            "Group complete source-bound vertex-deletion cards by exact graph "
            "isomorphism, retaining a representative, exact multiplicity, and "
            "source-card indices. Admits at most 10 source vertices and "
            "2000000 exact permutation canonicalization work units."
        ),
        request_type=UnlabelledVertexDeckRequest,
        result_type=UnlabelledVertexDeck,
        run=_run_unlabelled_vertex,
        tags=("graph", "deck", "isomorphism", "multiset", "exact"),
        discovery_terms=("unlabelled vertex deck", "vertex deck multiplicities"),
        examples=(
            OperationExample(
                name="path_vertex_quotient",
                description="P3 has two isomorphic endpoint-deleted cards and one distinct middle-deleted card.",
                input={
                    "deck": {
                        "source": {
                            "vertices": ["a", "b", "c"],
                            "edges": [["a", "b"], ["b", "c"]],
                        },
                        "cards": [
                            {
                                "deleted_vertex": "a",
                                "card": {"vertices": ["b", "c"], "edges": [["b", "c"]]},
                                "retained_vertices": ["b", "c"],
                                "retained_edge_count": 1,
                                "deleted_edge_count": 1,
                            },
                            {
                                "deleted_vertex": "b",
                                "card": {"vertices": ["a", "c"], "edges": []},
                                "retained_vertices": ["a", "c"],
                                "retained_edge_count": 0,
                                "deleted_edge_count": 2,
                            },
                            {
                                "deleted_vertex": "c",
                                "card": {"vertices": ["a", "b"], "edges": [["a", "b"]]},
                                "retained_vertices": ["a", "b"],
                                "retained_edge_count": 1,
                                "deleted_edge_count": 1,
                            },
                        ],
                        "edge_appearances": [1, 1],
                        "vertex_appearances": [2, 2, 2],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="graph.deck.vertex.edge_count.compute",
        title="Reconstruct edge count from a vertex deck",
        description=(
            "For a complete source-bound vertex deck of order n >= 3, calculate "
            "the multiset of card edge counts and recover the source edge count "
            "from sum_v |E(G-v)| = (n-2)|E(G)|. The operation validates the exact "
            "card quotient and multiplicities, then checks divisibility before "
            "returning the source count."
        ),
        request_type=VertexDeckEdgeCountRequest,
        result_type=VertexDeckEdgeCount,
        run=_run_vertex_deck_edge_count,
        tags=("graph", "deck", "vertex-deletion", "reconstruction", "exact"),
        discovery_terms=(
            "reconstruct graph edge count from vertex deck",
            "Kelly edge count identity",
            "vertex deck card edge counts",
        ),
        examples=(
            OperationExample(
                name="triangle_edge_count",
                description="Every vertex-deleted card of K3 has one edge, so the source has 3 edges.",
                input={
                    "deck": {
                        "family": {
                            "source": {
                                "vertices": ["a", "b", "c"],
                                "edges": [["a", "b"], ["a", "c"], ["b", "c"]],
                            },
                            "cards": [
                                {
                                    "deleted_vertex": "a",
                                    "card": {
                                        "vertices": ["b", "c"],
                                        "edges": [["b", "c"]],
                                    },
                                    "retained_vertices": ["b", "c"],
                                    "retained_edge_count": 1,
                                    "deleted_edge_count": 2,
                                },
                                {
                                    "deleted_vertex": "b",
                                    "card": {
                                        "vertices": ["a", "c"],
                                        "edges": [["a", "c"]],
                                    },
                                    "retained_vertices": ["a", "c"],
                                    "retained_edge_count": 1,
                                    "deleted_edge_count": 2,
                                },
                                {
                                    "deleted_vertex": "c",
                                    "card": {
                                        "vertices": ["a", "b"],
                                        "edges": [["a", "b"]],
                                    },
                                    "retained_vertices": ["a", "b"],
                                    "retained_edge_count": 1,
                                    "deleted_edge_count": 2,
                                },
                            ],
                            "edge_appearances": [1, 1, 1],
                            "vertex_appearances": [2, 2, 2],
                        },
                        "classes": [
                            {
                                "representative": {
                                    "vertices": ["b", "c"],
                                    "edges": [["b", "c"]],
                                },
                                "multiplicity": 3,
                                "card_indices": [0, 1, 2],
                            }
                        ],
                        "card_count": 3,
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="graph.deck.vertex.degree_multiset.compute",
        title="Reconstruct the vertex degree multiset from a vertex deck",
        description=(
            "For a complete source-bound unlabelled vertex deck of order n >= 3, "
            "derive the source edge count from the card edge-count identity and "
            "return the descending multiset (m-|E(C)|) over cards C. Uses the "
            "exact DegreeSequence value and admits/authenticates the bounded "
            "deck quotient before trusting card multiplicities. It returns no "
            "source-vertex labels or reconstructed graph."
        ),
        request_type=VertexDeckDegreeMultisetRequest,
        result_type=DegreeSequence,
        run=_run_vertex_deck_degree_multiset,
        tags=("graph", "deck", "degree-sequence", "reconstruction", "exact"),
        discovery_terms=(
            "degree multiset from vertex deck",
            "reconstruct graph degree sequence",
            "Kelly degree sequence",
        ),
        examples=(
            OperationExample(
                name="empty_graph_on_three_vertices",
                description=(
                    "Three empty two-vertex cards recover the zero degree "
                    "multiset, preserving all three repeated cards."
                ),
                input={
                    "deck": {
                        "family": {
                            "source": {"vertices": ["a", "b", "c"], "edges": []},
                            "cards": [
                                {
                                    "deleted_vertex": "a",
                                    "card": {"vertices": ["b", "c"], "edges": []},
                                    "retained_vertices": ["b", "c"],
                                    "retained_edge_count": 0,
                                    "deleted_edge_count": 0,
                                },
                                {
                                    "deleted_vertex": "b",
                                    "card": {"vertices": ["a", "c"], "edges": []},
                                    "retained_vertices": ["a", "c"],
                                    "retained_edge_count": 0,
                                    "deleted_edge_count": 0,
                                },
                                {
                                    "deleted_vertex": "c",
                                    "card": {"vertices": ["a", "b"], "edges": []},
                                    "retained_vertices": ["a", "b"],
                                    "retained_edge_count": 0,
                                    "deleted_edge_count": 0,
                                },
                            ],
                            "edge_appearances": [],
                            "vertex_appearances": [2, 2, 2],
                        },
                        "classes": [
                            {
                                "representative": {
                                    "vertices": ["b", "c"],
                                    "edges": [],
                                },
                                "multiplicity": 3,
                                "card_indices": [0, 1, 2],
                            }
                        ],
                        "card_count": 3,
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="graph.deck.vertex.induced_subgraph_count.compute",
        title="Reconstruct an induced pattern count from a vertex deck",
        description=(
            "For a pattern H with h < n source vertices, count induced vertex "
            "subsets isomorphic to H using Kelly's exact identity: the sum of "
            "card counts is (n-h) times the source count. Each card pattern count "
            "uses graph.induced_vertex_subset_pattern.count semantics (subsets, "
            "not labelled embeddings). Requires a complete exact vertex deck; "
            "preflights deck validation, canonicalization, all card-count work, "
            f"the result's label echo allocation, and a "
            f"{MAX_KELLY_DECK_TOTAL_WORK:,}-unit aggregate bound."
        ),
        request_type=VertexDeckInducedSubgraphCountRequest,
        result_type=VertexDeckInducedSubgraphCount,
        run=_run_vertex_deck_induced_pattern_count,
        tags=("graph", "deck", "Kelly-lemma", "induced-subgraph", "count", "exact"),
        discovery_terms=(
            "Kelly lemma induced pattern count",
            "reconstructible induced subgraph count",
            "vertex-deck pattern count",
        ),
        examples=(
            OperationExample(
                name="single_vertex_count_from_path_deck",
                description=(
                    "Recover the number of one-vertex induced subsets in P3. "
                    "Its three two-vertex cards contribute six total; divide "
                    "by n-h=2 to obtain three."
                ),
                input={
                    "deck": {
                        "family": {
                            "source": {
                                "vertices": ["a", "b", "c"],
                                "edges": [["a", "b"], ["b", "c"]],
                            },
                            "cards": [
                                {
                                    "deleted_vertex": "a",
                                    "card": {
                                        "vertices": ["b", "c"],
                                        "edges": [["b", "c"]],
                                    },
                                    "retained_vertices": ["b", "c"],
                                    "retained_edge_count": 1,
                                    "deleted_edge_count": 1,
                                },
                                {
                                    "deleted_vertex": "b",
                                    "card": {"vertices": ["a", "c"], "edges": []},
                                    "retained_vertices": ["a", "c"],
                                    "retained_edge_count": 0,
                                    "deleted_edge_count": 2,
                                },
                                {
                                    "deleted_vertex": "c",
                                    "card": {
                                        "vertices": ["a", "b"],
                                        "edges": [["a", "b"]],
                                    },
                                    "retained_vertices": ["a", "b"],
                                    "retained_edge_count": 1,
                                    "deleted_edge_count": 1,
                                },
                            ],
                            "edge_appearances": [1, 1],
                            "vertex_appearances": [2, 2, 2],
                        },
                        "classes": [
                            {
                                "representative": {
                                    "vertices": ["b", "c"],
                                    "edges": [["b", "c"]],
                                },
                                "multiplicity": 2,
                                "card_indices": [0, 2],
                            },
                            {
                                "representative": {"vertices": ["a", "c"], "edges": []},
                                "multiplicity": 1,
                                "card_indices": [1],
                            },
                        ],
                        "card_count": 3,
                    },
                    "pattern": {"vertices": ["x"], "edges": []},
                },
            ),
        ),
    ),
    MathTool(
        operation_id="graph.deck.vertex.subgraph_count.compute",
        title="Reconstruct an ordinary subgraph count from a vertex deck",
        description=(
            "For a proper simple graph pattern H with h < n vertices, count "
            "copies as vertex-subset and edge-subset pairs isomorphic to H; "
            "extra source edges on those vertices are allowed. The result is "
            "not an induced subset count and not an injective embedding count. "
            "Kelly's identity sums exact per-card copy counts with deck-class "
            "multiplicity and divides by n-h. Exact assignment, canonicalization, "
            "family, and result echo allocation work are admitted first, within "
            f"{MAX_KELLY_DECK_TOTAL_WORK:,} units."
        ),
        request_type=VertexDeckSubgraphCountRequest,
        result_type=VertexDeckSubgraphCount,
        run=_run_vertex_deck_subgraph_count,
        tags=("graph", "deck", "Kelly-lemma", "subgraph-count", "exact"),
        discovery_terms=(
            "ordinary subgraph count from vertex deck",
            "non-induced pattern copies",
            "Kelly subgraph counting lemma",
        ),
        examples=(
            OperationExample(
                name="edge_copies_from_path_deck",
                description=(
                    "P3 has two ordinary edge copies. Its three vertex-deleted "
                    "cards contain four edge copies total, so divide by n-h=2."
                ),
                input={
                    "deck": {
                        "family": {
                            "source": {
                                "vertices": ["a", "b", "c"],
                                "edges": [["a", "b"], ["b", "c"]],
                            },
                            "cards": [
                                {
                                    "deleted_vertex": "a",
                                    "card": {
                                        "vertices": ["b", "c"],
                                        "edges": [["b", "c"]],
                                    },
                                    "retained_vertices": ["b", "c"],
                                    "retained_edge_count": 1,
                                    "deleted_edge_count": 1,
                                },
                                {
                                    "deleted_vertex": "b",
                                    "card": {"vertices": ["a", "c"], "edges": []},
                                    "retained_vertices": ["a", "c"],
                                    "retained_edge_count": 0,
                                    "deleted_edge_count": 2,
                                },
                                {
                                    "deleted_vertex": "c",
                                    "card": {
                                        "vertices": ["a", "b"],
                                        "edges": [["a", "b"]],
                                    },
                                    "retained_vertices": ["a", "b"],
                                    "retained_edge_count": 1,
                                    "deleted_edge_count": 1,
                                },
                            ],
                            "edge_appearances": [1, 1],
                            "vertex_appearances": [2, 2, 2],
                        },
                        "classes": [
                            {
                                "representative": {
                                    "vertices": ["b", "c"],
                                    "edges": [["b", "c"]],
                                },
                                "multiplicity": 2,
                                "card_indices": [0, 2],
                            },
                            {
                                "representative": {"vertices": ["a", "c"], "edges": []},
                                "multiplicity": 1,
                                "card_indices": [1],
                            },
                        ],
                        "card_count": 3,
                    },
                    "pattern": {"vertices": ["x", "y"], "edges": [["x", "y"]]},
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
