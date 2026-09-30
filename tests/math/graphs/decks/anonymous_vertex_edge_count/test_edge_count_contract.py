"""Regressions for the anonymous vertex-deck edge-count public contract.

The result field names the presupposition it carries: the edge count any
source graph *realizing* the deck must have. The operation does not establish
that such a graph exists, and the published field, description, and example
must all say so.
"""

from __future__ import annotations

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.math.graphs.decks._models import (
    AnonymousGraphCardMultiset,
)
from jacobian.math.graphs.decks.anonymous_vertex_edge_count import (
    AnonymousVertexDeckEdgeCount,
    anonymous_vertex_deck_edge_count,
)
from jacobian.math.graphs.decks.anonymous_vertex_edge_count._tools import TOOLS
from jacobian.math.graphs.values import SimpleUndirectedGraph

OPERATION_ID = "graph.deck.anonymous_vertex.edge_count.compute"


CARD_LABELS = ("v00", "v01", "v02")


def _nonrealizable_deck() -> AnonymousGraphCardMultiset:
    """Two edgeless cards and two two-edge paths, on card order three.

    This is the reviewer's example. No four-vertex graph with two edges has
    card edge-count multiset ``[0, 0, 2, 2]``: the only possibilities are
    ``[1, 1, 1, 1]`` for a matching and ``[0, 1, 1, 2]`` for adjacent edges.
    The deck is structurally valid, so it is admitted, and the quotient
    ``(0*2 + 2*2) / (4-2) = 2`` is still returned.
    """
    edgeless = SimpleUndirectedGraph(vertices=CARD_LABELS, edges=())
    path = SimpleUndirectedGraph(
        vertices=CARD_LABELS, edges=(("v00", "v01"), ("v01", "v02"))
    )
    return AnonymousGraphCardMultiset.model_validate(
        {
            "card_order": 3,
            "classes": [
                {"representative": edgeless.model_dump(), "multiplicity": 2},
                {"representative": path.model_dump(), "multiplicity": 2},
            ],
        }
    )


# --- the public field names the presupposition it carries -------------------


def test_result_field_is_named_for_the_presupposition() -> None:
    fields = AnonymousVertexDeckEdgeCount.model_fields
    assert "realizing_edge_count" in fields
    assert "implied_edge_count" not in fields
    description = fields["realizing_edge_count"].description
    assert description is not None
    assert "does not establish" in description
    assert "realizab" in description


def test_result_carries_no_implied_edge_count_field() -> None:
    deck = _nonrealizable_deck()
    result = anonymous_vertex_deck_edge_count(deck)
    assert "implied_edge_count" not in type(result).model_fields
    assert "realizing_edge_count" in result.model_dump()


# --- a nonrealizable deck still returns the necessary quotient ---------------


def test_nonrealizable_deck_returns_the_necessary_quotient() -> None:
    """The quotient is a necessary consequence, not a realizability decision."""
    result = anonymous_vertex_deck_edge_count(_nonrealizable_deck())
    assert result.source_order == 4
    assert result.overcount_divisor == 2
    assert result.realizing_edge_count == 2


# --- the published description states the presupposition --------------------


def test_published_description_states_the_presupposition() -> None:
    tool = next(item for item in TOOLS if item.operation_id == OPERATION_ID)
    assert "realizing_edge_count" in tool.description
    assert "whether or not such a graph exists" in tool.description


def test_published_example_states_the_presupposition() -> None:
    tool = next(item for item in TOOLS if item.operation_id == OPERATION_ID)
    assert "does not itself establish realizability" in tool.examples[0].description


# --- the operation stays discoverable ---------------------------------------


def test_operation_remains_published_and_discoverable() -> None:
    """Kept published: the field and description now carry the caveat."""
    operation = Catalog.open().operation(OPERATION_ID)
    assert operation is not None
    assert operation.operation_id == OPERATION_ID


# --- negative controls -----------------------------------------------------


def test_realizable_deck_still_matches_its_source_edge_count() -> None:
    """P3 is realizable, so the quotient is genuinely its two source edges."""
    from jacobian.math.graphs.decks.operations import anonymous_graph_card_multiset

    vertices = ("v0", "v1", "v2")
    source = SimpleUndirectedGraph(
        vertices=vertices, edges=(("v0", "v1"), ("v1", "v2"))
    )
    cards = tuple(
        SimpleUndirectedGraph(
            vertices=tuple(v for v in vertices if v != deleted),
            edges=tuple(edge for edge in source.edges if deleted not in edge),
        )
        for deleted in vertices
    )
    deck = anonymous_graph_card_multiset(card_order=2, cards=cards)
    assert anonymous_vertex_deck_edge_count(deck).realizing_edge_count == len(
        source.edges
    )


def test_nondivisible_card_total_is_still_refused() -> None:
    """Divisibility remains a genuine rejection, not a softened one."""
    from jacobian.catalog.models import OperationDomainValidationError

    # Four cards, one of which carries a single edge, so the card edge total is
    # 1, which is not divisible by the Kelly divisor of 2.
    deck = AnonymousGraphCardMultiset.model_validate(
        {
            "card_order": 3,
            "classes": [
                {
                    "representative": SimpleUndirectedGraph(
                        vertices=CARD_LABELS, edges=()
                    ).model_dump(),
                    "multiplicity": 3,
                },
                {
                    "representative": SimpleUndirectedGraph(
                        vertices=CARD_LABELS, edges=(("v00", "v01"),)
                    ).model_dump(),
                    "multiplicity": 1,
                },
            ],
        }
    )
    with pytest.raises(OperationDomainValidationError) as error:
        anonymous_vertex_deck_edge_count(deck)
    assert (
        str(error.value.errors()[0]["type"])
        == "graph_deck.anonymous_edge_count_nondivisible"
    )


def test_a_forged_quotient_is_still_refused() -> None:
    """The result validator still checks the structure it publishes."""
    deck = _nonrealizable_deck()
    result = anonymous_vertex_deck_edge_count(deck)
    with pytest.raises(ValueError):
        AnonymousVertexDeckEdgeCount.model_validate(
            {**result.model_dump(), "realizing_edge_count": 99}
        )
