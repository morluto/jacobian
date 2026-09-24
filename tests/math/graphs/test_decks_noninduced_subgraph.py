"""Kelly reconstruction for ordinary, non-induced graph copies."""

from __future__ import annotations

from itertools import combinations, permutations

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.graphs.decks import (
    unlabelled_vertex_deck,
    vertex_deck_subgraph_count,
    vertex_deletion_family,
)
from jacobian.math.graphs.decks._models import VertexDeckSubgraphCountRequest
from jacobian.math.graphs.decks._tools import TOOLS
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _all_graphs(order: int, prefix: str) -> tuple[SimpleUndirectedGraph, ...]:
    vertices = tuple(f"{prefix}{i}" for i in range(order))
    possible = tuple(combinations(vertices, 2))
    return tuple(
        SimpleUndirectedGraph(
            vertices=vertices,
            edges=tuple(edge for i, edge in enumerate(possible) if mask & (1 << i)),
        )
        for mask in range(1 << len(possible))
    )


def _ordinary_copy_oracle(
    host: SimpleUndirectedGraph, pattern: SimpleUndirectedGraph
) -> int:
    """Enumerate vertex and edge subsets directly, with no embedding quotient."""
    k = len(pattern.vertices)
    required_edges = len(pattern.edges)
    host_edge_set = {frozenset(edge) for edge in host.edges}
    pattern_edge_set = {frozenset(edge) for edge in pattern.edges}
    result = 0
    for chosen_vertices in combinations(host.vertices, k):
        candidate_edges = tuple(
            edge
            for edge in combinations(chosen_vertices, 2)
            if frozenset(edge) in host_edge_set
        )
        for chosen_edges in combinations(candidate_edges, required_edges):
            chosen_set = {frozenset(edge) for edge in chosen_edges}
            if any(
                {
                    frozenset((mapping[i], mapping[j]))
                    for i, j in combinations(range(k), 2)
                    if frozenset((pattern.vertices[i], pattern.vertices[j]))
                    in pattern_edge_set
                }
                == chosen_set
                for mapping in permutations(chosen_vertices)
            ):
                result += 1
    return result


def test_kelly_ordinary_copy_count_matches_exhaustive_edge_subset_oracle() -> None:
    for source_order in range(1, 5):
        patterns_by_order = {
            order: _all_graphs(order, f"p{order}_") for order in range(source_order)
        }
        for source in _all_graphs(source_order, "g"):
            deck = unlabelled_vertex_deck(vertex_deletion_family(source))
            for _pattern_order, patterns in patterns_by_order.items():
                for pattern in patterns:
                    result = vertex_deck_subgraph_count(deck, pattern)
                    assert result.occurrence_count == _ordinary_copy_oracle(
                        source, pattern
                    )
                    assert (
                        type(result).model_validate_json(result.model_dump_json())
                        == result
                    )


def test_noninduced_copies_are_not_embeddings_or_induced_copies() -> None:
    source = SimpleUndirectedGraph(
        vertices=("a", "b", "c", "d"),
        edges=(
            ("a", "b"),
            ("a", "c"),
            ("a", "d"),
            ("b", "c"),
            ("b", "d"),
            ("c", "d"),
        ),
    )
    path = SimpleUndirectedGraph(
        vertices=("x", "y", "z"), edges=(("x", "y"), ("y", "z"))
    )
    result = vertex_deck_subgraph_count(
        unlabelled_vertex_deck(vertex_deletion_family(source)), path
    )
    # Each 3-set induces K3: there are three edge-subset copies of P3,
    # six labelled embeddings, and no induced copy. K4 has four 3-sets.
    assert result.occurrence_count == 12
    assert result.weighted_card_total == 12
    assert result.overcount_divisor == 1


def test_catalog_declares_source_bound_ordinary_copy_operation() -> None:
    operation = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "graph.deck.vertex.subgraph_count.compute"
    )
    request = VertexDeckSubgraphCountRequest.model_validate(operation.examples[0].input)
    result = operation.run(request)
    assert result.occurrence_count == 2
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_pattern_must_have_strictly_fewer_vertices_than_source() -> None:
    source = SimpleUndirectedGraph(vertices=("a", "b"), edges=(("a", "b"),))
    deck = unlabelled_vertex_deck(vertex_deletion_family(source))
    pattern = SimpleUndirectedGraph(vertices=("x", "y"), edges=(("x", "y"),))
    with pytest.raises(OperationDomainValidationError, match="strictly smaller"):
        vertex_deck_subgraph_count(deck, pattern)


def test_work_admission_precedes_copy_count_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.graphs.decks.operations as operations

    source = SimpleUndirectedGraph(
        vertices=("a", "b", "c"), edges=(("a", "b"), ("b", "c"))
    )
    deck = unlabelled_vertex_deck(vertex_deletion_family(source))
    pattern = SimpleUndirectedGraph(vertices=("x",), edges=())

    def fail(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("copy expansion ran before aggregate admission")

    monkeypatch.setattr(operations, "MAX_KELLY_DECK_TOTAL_WORK", 0)
    monkeypatch.setattr(operations, "_noninduced_copy_count", fail)
    with pytest.raises(OperationResourceAdmissionError, match="total work bound"):
        vertex_deck_subgraph_count(deck, pattern)
