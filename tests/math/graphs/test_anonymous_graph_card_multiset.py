from __future__ import annotations

from math import comb, factorial
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.graphs.decks import (
    AnonymousGraphCardMultiset,
    anonymous_graph_card_multiset,
)
from jacobian.math.graphs.decks import operations as deck_operations
from jacobian.math.graphs.decks._models import AnonymousGraphCardMultisetRequest
from jacobian.math.graphs.values import SimpleUndirectedGraph


def graph(
    vertices: tuple[str, ...], edges: tuple[tuple[str, str], ...]
) -> SimpleUndirectedGraph:
    return SimpleUndirectedGraph(vertices=vertices, edges=edges)


def test_independent_relabelings_have_one_canonical_class_and_preserve_multiplicity() -> (
    None
):
    first = graph(("a", "b", "c", "d"), (("a", "b"), ("b", "c")))
    second = graph(("w", "x", "y", "z"), (("x", "z"), ("w", "x")))
    result = anonymous_graph_card_multiset(4, (first, second))
    assert len(result.classes) == 1
    assert result.classes[0].multiplicity == 2
    assert result.classes[0].representative.vertices == ("v00", "v01", "v02", "v03")
    assert result.classes[0].representative.edges == (("v01", "v03"), ("v02", "v03"))


def test_nonisomorphic_cards_and_multiplicity_are_distinguished() -> None:
    edge = graph(("a", "b", "c"), (("a", "b"),))
    path = graph(("x", "y", "z"), (("x", "y"), ("y", "z")))
    result = anonymous_graph_card_multiset(3, (edge, edge, path))
    assert tuple(item.multiplicity for item in result.classes) == (1, 2)
    assert tuple(item.representative.edges for item in result.classes) == tuple(
        sorted(item.representative.edges for item in result.classes)
    )


def test_empty_multiset_retains_explicit_card_order() -> None:
    result = anonymous_graph_card_multiset(5, ())
    assert result.card_order == 5
    assert result.classes == ()


def test_serialized_result_round_trips_and_preserves_canonical_identity() -> None:
    request = AnonymousGraphCardMultisetRequest(
        card_order=3,
        cards=(graph(("a", "b", "c"), (("a", "b"),)),),
    )
    result = anonymous_graph_card_multiset(
        getattr(request, "card_order", None), request.cards
    )
    restored = AnonymousGraphCardMultiset.model_validate_json(result.model_dump_json())
    assert restored == result


def test_deserialization_keeps_representative_validation_structural() -> None:
    result = AnonymousGraphCardMultiset.model_validate(
        {
            "card_order": 3,
            "classes": [
                {
                    "representative": {
                        "vertices": ["v00", "v01", "v02"],
                        "edges": [["v00", "v01"]],
                    },
                    "multiplicity": 1,
                }
            ],
        }
    )
    assert result.classes[0].representative.edges == (("v00", "v01"),)


def test_multiplicity_schema_matches_runtime_bounds_and_anchoring() -> None:
    schema = AnonymousGraphCardMultiset.model_json_schema()
    multiplicity = schema["$defs"]["AnonymousGraphCardClass"]["properties"][
        "multiplicity"
    ]
    assert multiplicity["maxLength"] == 12
    assert multiplicity["pattern"] == r"^[1-9][0-9]{0,11}(?![\s\S])"
    validator = Draft202012Validator(schema)
    payload: dict[str, Any] = {
        "card_order": 0,
        "classes": [
            {
                "representative": {"vertices": [], "edges": []},
                "multiplicity": "1",
            }
        ],
    }
    assert validator.is_valid(payload)
    for value in ("0", "-1", "1000000000000", "1\n"):
        payload["classes"][0]["multiplicity"] = value
        assert not validator.is_valid(payload)
        with pytest.raises(ValidationError):
            AnonymousGraphCardMultiset.model_validate_json(
                '{"card_order":0,"classes":[{"representative":{"vertices":[],"edges":[]},"multiplicity":"'
                + value
                + '"}]}'
            )


def test_deserialization_rejects_duplicate_isomorphism_classes() -> None:
    one_edge = {"vertices": ["v00", "v01", "v02"], "edges": [["v01", "v02"]]}
    with pytest.raises(ValidationError, match="classes must be unique"):
        AnonymousGraphCardMultiset.model_validate(
            {
                "card_order": 3,
                "classes": [
                    {"representative": one_edge, "multiplicity": 1},
                    {"representative": one_edge, "multiplicity": 2},
                ],
            }
        )


def test_mixed_card_orders_are_rejected() -> None:
    with pytest.raises(ValidationError, match="declared card_order"):
        AnonymousGraphCardMultisetRequest(
            card_order=2,
            cards=(graph(("a", "b"), ()), graph(("x",), ())),
        )


@pytest.mark.parametrize("card_order", [None, True, "2", -1])
def test_malformed_constructed_card_order_is_a_domain_error(
    card_order: object,
) -> None:
    request = (
        AnonymousGraphCardMultisetRequest.model_construct(cards=())
        if card_order is None
        else AnonymousGraphCardMultisetRequest.model_construct(
            card_order=card_order, cards=()
        )
    )
    with pytest.raises(OperationDomainValidationError) as exc_info:
        anonymous_graph_card_multiset(
            getattr(request, "card_order", None), request.cards
        )
    assert exc_info.value.errors()[0]["type"] == "graph_deck.anonymous_order_invalid"


def test_forged_catalog_request_missing_card_order_is_a_domain_error() -> None:
    tool = Catalog.open().operation("graph.deck.from_cards.construct")
    assert tool is not None
    forged = tool.request_type.model_construct(cards=())
    with pytest.raises(OperationDomainValidationError) as exc_info:
        tool.run(forged)
    assert exc_info.value.errors()[0]["type"] == "graph_deck.anonymous_order_invalid"


def test_card_order_above_supported_bound_is_resource_error() -> None:
    from jacobian.math.graphs.decks._models import MAX_UNLABELLED_DECK_VERTICES

    request = AnonymousGraphCardMultisetRequest.model_construct(
        card_order=MAX_UNLABELLED_DECK_VERTICES + 1, cards=()
    )
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        anonymous_graph_card_multiset(
            getattr(request, "card_order", None), request.cards
        )
    assert exc_info.value.errors()[0]["type"] == "graph_deck.anonymous_order_bound"


def test_model_construct_parent_and_nested_parent_attacks_are_rejected() -> None:
    class RequestChild(AnonymousGraphCardMultisetRequest):
        pass

    valid = graph(("a", "b"), (("a", "b"),))
    child_request = RequestChild.model_construct(card_order=2, cards=(valid,))
    assert anonymous_graph_card_multiset(
        child_request.card_order, child_request.cards
    ).classes

    forged_graph = SimpleUndirectedGraph.model_construct(
        vertices=("a", "b"), edges=(("a", "missing"),)
    )
    forged_request = AnonymousGraphCardMultisetRequest.model_construct(
        card_order=2, cards=(forged_graph,)
    )
    with pytest.raises(OperationDomainValidationError, match="declared vertices"):
        anonymous_graph_card_multiset(forged_request.card_order, forged_request.cards)


@pytest.mark.parametrize(
    ("vertices", "edges", "message"),
    [
        (("a", "b"), (("a", "b"),) * 2, "more edge entries"),
        (("a", "b"), (("a", "missing"),), "declared vertices"),
        (("a" * 65, "b"), (), "label bound"),
        (("😀" * 17, "b"), (), "byte bound"),
        (("e\u0301", "b"), (), "NFC"),
        (("\ud800", "b"), (), "Unicode scalar"),
    ],
)
def test_model_construct_graph_shape_label_and_edge_bounds(
    vertices: tuple[str, ...], edges: tuple[tuple[str, str], ...], message: str
) -> None:
    forged_graph = SimpleUndirectedGraph.model_construct(vertices=vertices, edges=edges)
    request = AnonymousGraphCardMultisetRequest.model_construct(
        card_order=2, cards=(forged_graph,)
    )
    with pytest.raises(OperationDomainValidationError, match=message):
        anonymous_graph_card_multiset(
            getattr(request, "card_order", None), request.cards
        )


def test_model_construct_duplicate_edges_are_rejected() -> None:
    forged_graph = SimpleUndirectedGraph.model_construct(
        vertices=("a", "b", "c"), edges=(("a", "b"), ("a", "b"))
    )
    request = AnonymousGraphCardMultisetRequest.model_construct(
        card_order=3, cards=(forged_graph,)
    )
    with pytest.raises(OperationDomainValidationError, match="unique canonical pairs"):
        anonymous_graph_card_multiset(
            getattr(request, "card_order", None), request.cards
        )


@pytest.mark.parametrize(
    ("endpoint", "message"),
    [("x" * 65, "within the graph label bound"), ("\ud800", "Unicode scalar")],
)
def test_model_construct_edge_endpoints_are_bounded_before_set_checks(
    endpoint: str, message: str
) -> None:
    forged_graph = SimpleUndirectedGraph.model_construct(
        vertices=("a", "b"), edges=((endpoint, "b"),)
    )
    request = AnonymousGraphCardMultisetRequest.model_construct(
        card_order=2, cards=(forged_graph,)
    )
    with pytest.raises(OperationDomainValidationError, match=message):
        anonymous_graph_card_multiset(
            getattr(request, "card_order", None), request.cards
        )


def test_permutation_bound_accepts_exact_limit_and_rejects_one_unit_less(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = graph(("a", "b", "c", "d"), (("a", "b"),))
    request = AnonymousGraphCardMultisetRequest(card_order=4, cards=(item,))
    exact_work = factorial(4) * (4 + 2 * comb(4, 2))
    monkeypatch.setattr(
        deck_operations,
        "MAX_ANONYMOUS_CARD_CANONICALIZATION_WORK",
        exact_work,
        raising=False,
    )
    assert anonymous_graph_card_multiset(
        getattr(request, "card_order", None), request.cards
    ).classes
    monkeypatch.setattr(
        deck_operations,
        "MAX_ANONYMOUS_CARD_CANONICALIZATION_WORK",
        exact_work - 1,
        raising=False,
    )
    with pytest.raises(OperationResourceAdmissionError, match="work bound"):
        anonymous_graph_card_multiset(
            getattr(request, "card_order", None), request.cards
        )


def test_tied_order_eight_candidates_pay_for_full_vector_comparison() -> None:
    vertices = tuple(f"v{i:02d}" for i in range(8))
    request = AnonymousGraphCardMultisetRequest(
        card_order=8, cards=(graph(vertices, ()),)
    )
    tied_work = factorial(8) * (8 + 2 * comb(8, 2))
    assert tied_work > 2_000_000
    with pytest.raises(OperationResourceAdmissionError, match="work bound"):
        anonymous_graph_card_multiset(
            getattr(request, "card_order", None), request.cards
        )

    payload = {
        "card_order": 8,
        "classes": [
            {
                "representative": {"vertices": list(vertices), "edges": []},
                "multiplicity": 1,
            }
        ],
    }
    decoded = AnonymousGraphCardMultiset.model_validate(payload)
    assert decoded.card_order == 8
    assert decoded.classes[0].representative.edges == ()


def test_catalog_publishes_anonymous_cards_as_distinct_from_realizable_decks() -> None:
    tool = Catalog.open().operation("graph.deck.from_cards.construct")
    assert tool is not None
    result = tool.run(
        tool.request_type.model_validate(
            {"card_order": 1, "cards": [{"vertices": ["x"], "edges": []}]}
        )
    )
    assert result.card_order == 1
    assert result.classes[0].multiplicity == 1
