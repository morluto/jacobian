from __future__ import annotations

from collections import Counter
from itertools import combinations, permutations
from math import comb

import pytest
from pydantic import ValidationError

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.graphs.decks import (
    AnonymousGraphCardMultiset,
    VertexDeckAnonymousMultisetRequest,
    vertex_deck_anonymous_multiset,
    vertex_deletion_family,
)
from jacobian.math.graphs.decks.anonymous_equality._models import (
    AnonymousDeckEqualityRequest,
)
from jacobian.math.graphs.decks.anonymous_equality.operations import (
    anonymous_deck_equality,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _graph(order: int, edge_mask: int) -> SimpleUndirectedGraph:
    vertices = tuple(f"x{i}" for i in range(order))
    pairs = tuple(combinations(vertices, 2))
    edges = tuple(edge for bit, edge in enumerate(pairs) if edge_mask & (1 << bit))
    return SimpleUndirectedGraph(vertices=vertices, edges=edges)


def _oracle_card_edges(graph: SimpleUndirectedGraph) -> tuple[tuple[str, str], ...]:
    """Independent full-permutation adjacency-vector oracle for small cards."""
    pairs = tuple(combinations(range(len(graph.vertices)), 2))
    index = {vertex: i for i, vertex in enumerate(graph.vertices)}
    edges = {frozenset((index[left], index[right])) for left, right in graph.edges}
    best: tuple[int, ...] | None = None
    for ordering in permutations(range(len(graph.vertices))):
        code = tuple(
            int(frozenset((ordering[i], ordering[j])) in edges) for i, j in pairs
        )
        if best is None or code < best:
            best = code
    canonical_vertices = tuple(f"v{i:02d}" for i in range(len(graph.vertices)))
    return tuple(
        (canonical_vertices[i], canonical_vertices[j])
        for bit, (i, j) in zip(best or (), pairs, strict=True)
        if bit
    )


def _expected_deck(
    graph: SimpleUndirectedGraph,
) -> tuple[int, Counter[tuple[tuple[str, str], ...]]]:
    card_order = max(len(graph.vertices) - 1, 0)
    counts: Counter[tuple[tuple[str, str], ...]] = Counter()
    family = vertex_deletion_family(graph)
    for card in family.cards:
        counts[_oracle_card_edges(card.card)] += 1
    return card_order, counts


def _actual_deck(
    value: AnonymousGraphCardMultiset,
) -> tuple[int, Counter[tuple[tuple[str, str], ...]]]:
    return value.card_order, Counter(
        {item.representative.edges: item.multiplicity for item in value.classes}
    )


@pytest.mark.parametrize(
    ("order", "edge_mask"),
    [
        (order, edge_mask)
        for order in range(5)
        for edge_mask in range(1 << comb(order, 2))
    ],
)
def test_source_bound_deck_maps_to_exact_anonymous_isomorphism_multiset(
    order: int, edge_mask: int
) -> None:
    source = _graph(order, edge_mask)
    result = vertex_deck_anonymous_multiset(
        VertexDeckAnonymousMultisetRequest(family=vertex_deletion_family(source))
    )

    assert _actual_deck(result) == _expected_deck(source)


def test_anonymous_result_composes_with_global_deck_equality() -> None:
    left_graph = SimpleUndirectedGraph(
        vertices=("a", "b", "c", "d"),
        edges=(("a", "b"), ("b", "c"), ("c", "d")),
    )
    relabelled = SimpleUndirectedGraph(
        vertices=("w", "x", "y", "z"),
        edges=(("w", "z"), ("x", "y"), ("y", "z")),
    )
    different = SimpleUndirectedGraph(
        vertices=("p", "q", "r", "s"),
        edges=(("p", "q"), ("p", "r"), ("p", "s")),
    )

    def anonymous(graph: SimpleUndirectedGraph) -> AnonymousGraphCardMultiset:
        return vertex_deck_anonymous_multiset(
            VertexDeckAnonymousMultisetRequest(family=vertex_deletion_family(graph))
        )

    left, same, other = map(anonymous, (left_graph, relabelled, different))
    assert anonymous_deck_equality(
        AnonymousDeckEqualityRequest(left=left, right=same)
    ).equal
    assert not anonymous_deck_equality(
        AnonymousDeckEqualityRequest(left=left, right=other)
    ).equal


def test_source_family_replay_rejects_forged_card_before_canonicalizing() -> None:
    graph = _graph(3, 3)
    family = vertex_deletion_family(graph)
    forged = family.model_copy(
        update={"cards": (family.cards[0], family.cards[0], family.cards[2])}
    )
    with pytest.raises(
        OperationDomainValidationError,
        match="every exact source vertex-deleted card",
    ):
        request = VertexDeckAnonymousMultisetRequest.model_construct(family=forged)
        vertex_deck_anonymous_multiset(request)


def test_canonicalization_bound_is_checked_before_source_family_replay() -> None:
    source = _graph(9, 0)
    family = vertex_deletion_family(source)
    request = VertexDeckAnonymousMultisetRequest(family=family)
    with pytest.raises(OperationResourceAdmissionError):
        vertex_deck_anonymous_multiset(request)


def test_raw_request_keeps_semantic_admission_in_operation_path() -> None:
    with pytest.raises(ValidationError, match="cards"):
        VertexDeckAnonymousMultisetRequest.model_validate(
            {
                "family": {
                    "source": {
                        "vertices": [f"v{i}" for i in range(9)],
                        "edges": [],
                    },
                    "cards": "malformed nested cards",
                }
            }
        )


def test_operation_is_published_with_typed_output_and_equality_example() -> None:
    declaration = Catalog.open().operation("graph.deck.vertex.anonymous.compute")
    assert declaration is not None
    assert declaration.result_type.__name__ == "AnonymousGraphCardMultiset"
    assert declaration.examples
