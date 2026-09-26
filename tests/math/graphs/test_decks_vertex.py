"""Tests for graph.deck.vertex_deleted.compute."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.graphs.decks._models import (
    UnlabelledVertexDeckClass,
    UnlabelledVertexDeckRequest,
    VertexDeckInducedSubgraphCountRequest,
    VertexDeckRequest,
    VertexDeletionFamily,
)
from jacobian.math.graphs.decks._tools import TOOLS, _run_vertex_deleted
from jacobian.math.graphs.decks.operations import (
    unlabelled_vertex_deck,
    verify_vertex_deletion_family,
    vertex_deck_induced_subgraph_count,
    vertex_deletion_family,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _graph(
    vertices: tuple[str, ...], edges: tuple[tuple[str, str], ...]
) -> SimpleUndirectedGraph:
    return SimpleUndirectedGraph(vertices=vertices, edges=edges)


def _path_3() -> SimpleUndirectedGraph:
    return _graph(("a", "b", "c"), (("a", "b"), ("b", "c")))


def test_known_answer_path_3() -> None:
    """P3 yields three cards with one retained edge each in total two."""
    result = vertex_deletion_family(_path_3())
    assert len(result.cards) == 3
    assert [card.deleted_vertex for card in result.cards] == ["a", "b", "c"]
    assert result.cards[0].card.vertices == ("b", "c")
    assert result.cards[0].card.edges == (("b", "c"),)
    assert result.cards[1].card.vertices == ("a", "c")
    assert result.cards[1].card.edges == ()
    assert result.cards[2].card.vertices == ("a", "b")
    assert result.edge_appearances == (1, 1)
    assert result.vertex_appearances == (2, 2, 2)


def test_cycle_4_edge_identity() -> None:
    """Every C4 edge appears in exactly n-2 = 2 cards."""
    graph = _graph(
        ("a", "b", "c", "d"),
        (("a", "b"), ("b", "c"), ("c", "d"), ("a", "d")),
    )
    result = vertex_deletion_family(graph)
    assert result.edge_appearances == (2, 2, 2, 2)
    assert result.vertex_appearances == (3, 3, 3, 3)
    assert sum(card.retained_edge_count for card in result.cards) == 2 * 4


def test_complete_graph_k4_accounting() -> None:
    """K4: each card is K3 and the (n-2)m identity holds."""
    vertices = ("a", "b", "c", "d")
    edges = tuple(
        (u, v) if u < v else (v, u)
        for i, u in enumerate(vertices)
        for v in vertices[i + 1 :]
    )
    graph = _graph(vertices, edges)
    result = vertex_deletion_family(graph)
    assert result.edge_appearances == (2,) * 6
    for card in result.cards:
        assert len(card.card.edges) == 3
        assert card.retained_edge_count == 3
        assert card.deleted_edge_count == 3


def test_boundary_single_vertex() -> None:
    """K1 yields one empty card; appearance ledgers are vacuous."""
    result = vertex_deletion_family(_graph(("a",), ()))
    assert len(result.cards) == 1
    assert result.cards[0].card.vertices == ()
    assert result.edge_appearances == ()
    assert result.vertex_appearances == (0,)


def test_boundary_empty_graph() -> None:
    """The null graph yields the empty family."""
    result = vertex_deletion_family(_graph((), ()))
    assert result.cards == ()
    assert result.edge_appearances == ()
    assert result.vertex_appearances == ()


def test_boundary_edgeless_graph() -> None:
    """Edgeless orders keep vertex accounting with no edge rows."""
    result = vertex_deletion_family(_graph(("a", "b", "c"), ()))
    assert result.edge_appearances == ()
    assert result.vertex_appearances == (2, 2, 2)
    for card in result.cards:
        assert card.card.edges == ()


def test_serialized_family_rejects_forged_card_and_receipt() -> None:
    result = vertex_deletion_family(_path_3())
    payload = json.loads(result.model_dump_json())
    payload["cards"][0]["card"]["edges"] = []
    with pytest.raises(ValidationError):
        VertexDeletionFamily.model_validate(payload)

    payload = json.loads(result.model_dump_json())
    payload["edge_appearances"][0] = 0
    with pytest.raises(ValidationError):
        VertexDeletionFamily.model_validate(payload)


def test_adversarial_dropped_card_fails_verify() -> None:
    """Omitting a card breaks coverage and fails verification."""
    result = vertex_deletion_family(_path_3())
    mutated = VertexDeletionFamily.model_construct(
        source=result.source,
        cards=result.cards[:2],
        edge_appearances=result.edge_appearances,
        vertex_appearances=result.vertex_appearances,
    )
    assert not verify_vertex_deletion_family(mutated)


def test_adversarial_rewired_card_fails_verify() -> None:
    """Adding a foreign edge to a card fails verification."""
    result = vertex_deletion_family(_path_3())
    first = result.cards[0]
    rewired_card = first.model_copy(
        update={
            "card": _graph(("b", "c"), (("b", "c"),)),
            "retained_edge_count": 5,
        }
    )
    mutated = VertexDeletionFamily.model_construct(
        source=result.source,
        cards=(rewired_card, *result.cards[1:]),
        edge_appearances=result.edge_appearances,
        vertex_appearances=result.vertex_appearances,
    )
    assert not verify_vertex_deletion_family(mutated)


def test_verify_accepts_true_claim() -> None:
    """The kernel output verifies against itself."""
    assert verify_vertex_deletion_family(vertex_deletion_family(_path_3()))


def test_defining_invariant_kelly_identities() -> None:
    """Aggregate identities hold on a mixed asymmetric fixture."""
    graph = _graph(
        ("a", "b", "c", "d", "e"),
        (("a", "b"), ("b", "c"), ("c", "d"), ("a", "d"), ("a", "c")),
    )
    result = vertex_deletion_family(graph)
    order = len(graph.vertices)
    assert all(count == order - 2 for count in result.edge_appearances)
    assert all(count == order - 1 for count in result.vertex_appearances)
    assert result.edge_appearances == tuple(
        sum(edge in card.card.edges for card in result.cards) for edge in graph.edges
    )
    assert result.vertex_appearances == tuple(
        sum(vertex in card.card.vertices for card in result.cards)
        for vertex in graph.vertices
    )
    assert sum(card.retained_edge_count for card in result.cards) == (order - 2) * len(
        graph.edges
    )
    for card in result.cards:
        assert card.retained_edge_count + card.deleted_edge_count == len(graph.edges)


def test_native_vs_catalog_parity() -> None:
    """The catalog wrapper runs the same admitted kernel as the native call."""
    request = VertexDeckRequest(graph=_path_3())
    assert _run_vertex_deleted(request) == vertex_deletion_family(request.graph)


def test_catalog_example_input_executes() -> None:
    """The published operation example is admitted and complete."""
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "graph.deck.vertex_deleted.compute"
    )
    assert tool.examples
    request = VertexDeckRequest.model_validate(tool.examples[0].input)
    result = tool.run(request)
    assert len(result.cards) == 3


def test_oversized_source_rejected_before_expansion() -> None:
    """Sources above the vertex envelope are refused without materialization."""
    vertices = tuple(f"v{i:03d}" for i in range(65))
    graph = _graph(vertices, ())
    with pytest.raises(OperationResourceAdmissionError):
        vertex_deletion_family(graph)


def test_vertex_deck_multiset_matches_exhaustive_permutation_oracle() -> None:
    """Every labeled source graph through order four matches NetworkX VF2."""
    from itertools import combinations

    import networkx as nx

    def as_networkx(graph: SimpleUndirectedGraph) -> nx.Graph:
        result = nx.Graph()
        result.add_nodes_from(graph.vertices)
        result.add_edges_from(graph.edges)
        return result

    for order in range(5):
        vertices = tuple(f"v{i}" for i in range(order))
        possible_edges = tuple(combinations(vertices, 2))
        for mask in range(1 << len(possible_edges)):
            edges = tuple(
                edge for bit, edge in enumerate(possible_edges) if mask & (1 << bit)
            )
            source = SimpleUndirectedGraph(vertices=vertices, edges=edges)
            family = vertex_deletion_family(source)
            deck = unlabelled_vertex_deck(family)
            expected: list[list[int]] = []
            expected_representatives: list[nx.Graph] = []
            for index, card in enumerate(family.cards):
                graph_card = as_networkx(card.card)
                for class_index, representative in enumerate(expected_representatives):
                    if nx.is_isomorphic(graph_card, representative):
                        expected[class_index].append(index)
                        break
                else:
                    expected_representatives.append(graph_card)
                    expected.append([index])
            assert deck.card_count == order
            assert [list(item.card_indices) for item in deck.classes] == expected
            assert all(
                item.multiplicity == len(item.card_indices) for item in deck.classes
            )
            assert sum(item.multiplicity for item in deck.classes) == order


def test_vertex_unlabelled_catalog_example_and_serialized_composition() -> None:
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "graph.deck.vertex.unlabelled.compute"
    )
    request = UnlabelledVertexDeckRequest.model_validate(tool.examples[0].input)
    result = tool.run(request)
    assert [item.multiplicity for item in result.classes] == [2, 1]
    decoded = type(result).model_validate_json(result.model_dump_json())
    assert decoded == result


def test_vertex_unlabelled_consumer_rejects_forged_family() -> None:
    family = vertex_deletion_family(_path_3())
    forged = VertexDeletionFamily.model_construct(
        source=family.source,
        cards=family.cards[:-1],
        edge_appearances=family.edge_appearances,
        vertex_appearances=family.vertex_appearances,
    )
    with pytest.raises(OperationDomainValidationError, match="complete source family"):
        unlabelled_vertex_deck(forged)


def test_vertex_unlabelled_exact_work_boundary() -> None:
    accepted = SimpleUndirectedGraph(
        vertices=tuple(f"v{i}" for i in range(8)), edges=()
    )
    result = unlabelled_vertex_deck(vertex_deletion_family(accepted))
    assert result.classes[0].multiplicity == 8

    rejected = SimpleUndirectedGraph(
        vertices=tuple(f"v{i}" for i in range(9)), edges=()
    )
    family = vertex_deletion_family(rejected)
    with pytest.raises(OperationResourceAdmissionError, match="permutation work"):
        unlabelled_vertex_deck(family)


def test_kelly_induced_count_matches_exhaustive_direct_graph_oracle() -> None:
    """Every host through order four and every proper pattern match direct subsets."""
    from itertools import combinations

    import networkx as nx

    def graph_from_mask(prefix: str, order: int, mask: int) -> SimpleUndirectedGraph:
        vertices = tuple(f"{prefix}{index}" for index in range(order))
        possible_edges = tuple(combinations(vertices, 2))
        return SimpleUndirectedGraph(
            vertices=vertices,
            edges=tuple(
                edge for bit, edge in enumerate(possible_edges) if mask & (1 << bit)
            ),
        )

    def networkx_graph(graph: SimpleUndirectedGraph) -> nx.Graph:
        result = nx.Graph()
        result.add_nodes_from(graph.vertices)
        result.add_edges_from(graph.edges)
        return result

    def direct_induced_count(
        host: SimpleUndirectedGraph, pattern: SimpleUndirectedGraph
    ) -> int:
        host_nx = networkx_graph(host)
        pattern_nx = networkx_graph(pattern)
        return sum(
            nx.is_isomorphic(host_nx.subgraph(subset), pattern_nx)
            for subset in combinations(host.vertices, len(pattern.vertices))
        )

    for source_order in range(1, 5):
        source_edge_count = source_order * (source_order - 1) // 2
        for source_mask in range(1 << source_edge_count):
            source = graph_from_mask("g", source_order, source_mask)
            family = vertex_deletion_family(source)
            deck = unlabelled_vertex_deck(family)
            for pattern_order in range(source_order):
                pattern_edge_count = pattern_order * (pattern_order - 1) // 2
                for pattern_mask in range(1 << pattern_edge_count):
                    pattern = graph_from_mask("p", pattern_order, pattern_mask)
                    result = vertex_deck_induced_subgraph_count(deck, pattern)
                    expected = direct_induced_count(source, pattern)
                    assert result.occurrence_count == expected
                    assert result.overcount_divisor == source_order - pattern_order
                    assert result.weighted_card_total == sum(
                        direct_induced_count(card.card, pattern)
                        for card in family.cards
                    )


def test_kelly_operation_example_and_proper_order_boundary() -> None:
    from jacobian.math.graphs.decks._tools import TOOLS as DECK_TOOLS

    tool = next(
        tool
        for tool in DECK_TOOLS
        if tool.operation_id == "graph.deck.vertex.induced_subgraph_count.compute"
    )
    request = VertexDeckInducedSubgraphCountRequest.model_validate(
        tool.examples[0].input
    )
    result = tool.run(request)
    assert result.occurrence_count == 3
    assert result.weighted_card_total == 6
    assert result.overcount_divisor == 2
    assert type(result).model_validate_json(result.model_dump_json()) == result

    deck = unlabelled_vertex_deck(
        vertex_deletion_family(
            SimpleUndirectedGraph(vertices=("a", "b"), edges=(("a", "b"),))
        )
    )
    with pytest.raises(ValidationError, match="strictly smaller"):
        VertexDeckInducedSubgraphCountRequest(
            deck=deck,
            pattern=SimpleUndirectedGraph(vertices=("x", "y"), edges=(("x", "y"),)),
        )


def test_kelly_induced_count_rejects_oversized_label_echo_before_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unbounded vertex labels must not expand into an oversized deck echo."""
    import jacobian.math.graphs.decks.operations as operations

    big = "x" * 200_000
    source = _graph((big, "b", "c"), (("b", "c"), ("b", big), ("c", big)))
    deck = unlabelled_vertex_deck(vertex_deletion_family(source))
    pattern = _graph(("p",), ())

    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("result expansion ran before the echo admission gate")

    monkeypatch.setattr(operations, "unlabelled_vertex_deck", fail)
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        vertex_deck_induced_subgraph_count(deck, pattern)
    assert (
        exc_info.value.errors()[0]["type"] == "graph_deck.kelly_result_allocation_bound"
    )


def test_kelly_induced_count_admits_bounded_label_echo() -> None:
    big = "x" * 2_000
    source = _graph((big, "b", "c"), (("b", "c"), ("b", big), ("c", big)))
    deck = unlabelled_vertex_deck(vertex_deletion_family(source))
    result = vertex_deck_induced_subgraph_count(deck, _graph(("p",), ()))
    assert result.occurrence_count == 3
    assert result.overcount_divisor == 2


def test_kelly_count_rejects_forged_deck_class_multiplicity() -> None:
    source = _path_3()
    deck = unlabelled_vertex_deck(vertex_deletion_family(source))
    first = deck.classes[0]
    forged_first = UnlabelledVertexDeckClass.model_construct(
        representative=first.representative,
        multiplicity=1,
        card_indices=(first.card_indices[0],),
    )
    forged = type(deck).model_construct(
        family=deck.family,
        classes=(forged_first, *deck.classes[1:]),
        card_count=deck.card_count,
    )
    pattern = SimpleUndirectedGraph(vertices=("x",), edges=())
    with pytest.raises(OperationDomainValidationError, match="exact multiset quotient"):
        vertex_deck_induced_subgraph_count(forged, pattern)
