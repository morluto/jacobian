from __future__ import annotations

from math import comb, factorial

import pytest
from pydantic import ValidationError

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.graphs.decks import (
    AnonymousGraphCardMultisetRequest,
    anonymous_graph_card_multiset,
)
from jacobian.math.graphs.decks import operations as deck_operations
from jacobian.math.graphs.values import SimpleUndirectedGraph


def graph(vertices: tuple[str, ...], edges: tuple[tuple[str, str], ...]) -> SimpleUndirectedGraph:
    return SimpleUndirectedGraph(vertices=vertices, edges=edges)


def test_independent_relabelings_have_one_canonical_class_and_preserve_multiplicity() -> None:
    first = graph(("a", "b", "c", "d"), (("a", "b"), ("b", "c")))
    second = graph(("w", "x", "y", "z"), (("x", "z"), ("w", "x")))
    result = anonymous_graph_card_multiset(
        AnonymousGraphCardMultisetRequest(card_order=4, cards=(first, second))
    )
    assert len(result.classes) == 1
    assert result.classes[0].multiplicity == 2
    assert result.classes[0].representative.vertices == ("v00", "v01", "v02", "v03")
    assert result.classes[0].representative.edges == (("v01", "v03"), ("v02", "v03"))


def test_nonisomorphic_cards_and_multiplicity_are_distinguished() -> None:
    edge = graph(("a", "b", "c"), (("a", "b"),))
    path = graph(("x", "y", "z"), (("x", "y"), ("y", "z")))
    result = anonymous_graph_card_multiset(
        AnonymousGraphCardMultisetRequest(card_order=3, cards=(edge, edge, path))
    )
    assert tuple(item.multiplicity for item in result.classes) == (1, 2)
    assert tuple(item.representative.edges for item in result.classes) == tuple(
        sorted(item.representative.edges for item in result.classes)
    )


def test_empty_multiset_retains_explicit_card_order() -> None:
    result = anonymous_graph_card_multiset(
        AnonymousGraphCardMultisetRequest(card_order=5, cards=())
    )
    assert result.card_order == 5
    assert result.classes == ()


def test_mixed_card_orders_are_rejected() -> None:
    with pytest.raises(ValidationError, match="declared card_order"):
        AnonymousGraphCardMultisetRequest(
            card_order=2,
            cards=(graph(("a", "b"), ()), graph(("x",), ())),
        )


def test_model_construct_parent_and_nested_parent_attacks_are_rejected() -> None:
    class RequestChild(AnonymousGraphCardMultisetRequest):
        pass

    valid = graph(("a", "b"), (("a", "b"),))
    child_request = RequestChild.model_construct(card_order=2, cards=(valid,))
    with pytest.raises(OperationDomainValidationError, match="AnonymousGraphCardMultisetRequest"):
        anonymous_graph_card_multiset(child_request)

    forged_graph = SimpleUndirectedGraph.model_construct(
        vertices=("a", "b"), edges=(("a", "missing"),)
    )
    forged_request = AnonymousGraphCardMultisetRequest.model_construct(
        card_order=2, cards=(forged_graph,)
    )
    with pytest.raises(OperationDomainValidationError, match="declared vertices"):
        anonymous_graph_card_multiset(forged_request)


def test_permutation_bound_accepts_exact_limit_and_rejects_one_unit_less(monkeypatch) -> None:
    item = graph(("a", "b", "c", "d"), (("a", "b"),))
    request = AnonymousGraphCardMultisetRequest(card_order=4, cards=(item,))
    exact_work = factorial(4) * (4 + comb(4, 2))
    monkeypatch.setattr(
        deck_operations, "MAX_ANONYMOUS_CARD_CANONICALIZATION_WORK", exact_work
    )
    assert anonymous_graph_card_multiset(request).classes
    monkeypatch.setattr(
        deck_operations, "MAX_ANONYMOUS_CARD_CANONICALIZATION_WORK", exact_work - 1
    )
    with pytest.raises(OperationResourceAdmissionError, match="work bound"):
        anonymous_graph_card_multiset(request)


def test_catalog_publishes_anonymous_cards_as_distinct_from_realizable_decks() -> None:
    tool = Catalog.open().operation("graph.deck.from_cards.construct")
    assert tool is not None
    result = tool.run(tool.request_type.model_validate(
        {"card_order": 1, "cards": [{"vertices": ["x"], "edges": []}]}
    ))
    assert result.card_order == 1
    assert result.classes[0].multiplicity == 1
