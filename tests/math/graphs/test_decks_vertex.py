"""Tests for graph.deck.vertex_deleted.compute."""

from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.graphs.decks._models import (
    VertexDeckRequest,
    VertexDeletionFamily,
)
from jacobian.math.graphs.decks._tools import TOOLS, _run_vertex_deleted
from jacobian.math.graphs.decks.operations import (
    verify_vertex_deletion_family,
    vertex_deletion_family,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _graph(vertices: tuple[str, ...], edges: tuple[tuple[str, str], ...]):
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
