import json
from itertools import combinations

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.graphs.decks._models import (
    AnonymousGraphCardClass,
    AnonymousGraphCardMultiset,
    AnonymousGraphCardMultisetRequest,
)
from jacobian.math.graphs.decks.card_component_profile._models import (
    AnonymousDeckComponentProfile,
    AnonymousDeckComponentProfileRequest,
)
from jacobian.math.graphs.decks.card_component_profile._tools import TOOLS
from jacobian.math.graphs.decks.card_component_profile.operations import (
    card_component_profile,
)
from jacobian.math.graphs.decks.operations import anonymous_graph_card_multiset
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _deck(cards: tuple[SimpleUndirectedGraph, ...]) -> AnonymousGraphCardMultiset:
    return anonymous_graph_card_multiset(
        AnonymousGraphCardMultisetRequest(card_order=4, cards=cards)
    )


def _oracle_component_orders(
    graph: SimpleUndirectedGraph,
) -> tuple[int, ...]:
    labels = graph.vertices
    adjacency: dict[str, set[str]] = {vertex: set() for vertex in labels}
    for left, right in graph.edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    unseen = set(labels)
    sizes = []
    while unseen:
        root = unseen.pop()
        stack = [root]
        size = 1
        while stack:
            vertex = stack.pop()
            for neighbor in adjacency[vertex] & unseen:
                unseen.remove(neighbor)
                stack.append(neighbor)
                size += 1
        sizes.append(size)
    return tuple(sorted(sizes))


def test_exhaustive_order_four_graph_multiset_matches_independent_oracle() -> None:
    vertices = ("a", "b", "c", "d")
    possible_edges = tuple(combinations(vertices, 2))
    cards = tuple(
        SimpleUndirectedGraph(
            vertices=vertices,
            edges=tuple(
                edge for bit, edge in enumerate(possible_edges) if mask & (1 << bit)
            ),
        )
        for mask in range(1 << len(possible_edges))
    )
    deck = _deck(cards)
    result = card_component_profile(AnonymousDeckComponentProfileRequest(deck=deck))
    expected: dict[tuple[int, ...], int] = {}
    for graph in cards:
        profile = _oracle_component_orders(graph)
        expected[profile] = expected.get(profile, 0) + 1
    assert {
        item.component_orders: item.multiplicity for item in result.profiles
    } == expected
    assert result.card_count == 64
    assert sum(item.multiplicity for item in result.profiles) == len(cards)


def test_cardwise_profile_is_unchanged_by_independent_relabelling_and_order() -> None:
    first = SimpleUndirectedGraph(
        vertices=("a", "b", "c", "d"),
        edges=(("a", "b"), ("b", "c"), ("c", "d")),
    )
    relabelled = SimpleUndirectedGraph(
        vertices=("w", "x", "y", "z"),
        edges=(("w", "y"), ("x", "y"), ("x", "z")),
    )
    triangle_isolate = SimpleUndirectedGraph(
        vertices=("p", "q", "r", "s"),
        edges=(("p", "q"), ("p", "r"), ("q", "r")),
    )
    left = card_component_profile(
        AnonymousDeckComponentProfileRequest(deck=_deck((first, triangle_isolate)))
    )
    right = card_component_profile(
        AnonymousDeckComponentProfileRequest(deck=_deck((triangle_isolate, relabelled)))
    )
    assert left == right
    assert {item.component_orders: item.multiplicity for item in left.profiles} == {
        (1, 3): 1,
        (4,): 1,
    }


def test_zero_order_empty_deck_and_catalog_example() -> None:
    empty_deck = anonymous_graph_card_multiset(
        AnonymousGraphCardMultisetRequest(card_order=0, cards=())
    )
    result = card_component_profile(
        AnonymousDeckComponentProfileRequest(deck=empty_deck)
    )
    assert result.card_order == 0
    assert result.card_count == 0
    assert result.profiles == ()

    tool = next(
        item
        for item in TOOLS
        if item.operation_id == "graph.deck.card_component_profile.compute"
    )
    request = tool.request_type.model_validate_json(
        json.dumps(tool.examples[0].input), strict=True
    )
    published_result = tool.run(request)
    assert {
        item.component_orders: item.multiplicity for item in published_result.profiles
    } == {
        (1, 3): 1,
        (4,): 2,
    }


@pytest.mark.parametrize("order", (8, 9, 10))
def test_high_order_deck_is_profiled_without_permutation_canonicalization(
    order: int,
) -> None:
    # A single order-8 row exhausts the old factorial canonicalization budget,
    # yet component sizes need no canonical form, so these decks must be
    # admitted and profiled directly.
    vertices = tuple(f"v{i:02d}" for i in range(order))
    edges = tuple((vertices[i], vertices[i + 1]) for i in range(order - 1))
    graph = SimpleUndirectedGraph(vertices=vertices, edges=edges)
    deck = AnonymousGraphCardMultiset(
        card_order=order,
        classes=(AnonymousGraphCardClass(representative=graph, multiplicity=5),),
    )
    result = card_component_profile(AnonymousDeckComponentProfileRequest(deck=deck))
    assert {item.component_orders: item.multiplicity for item in result.profiles} == {
        (order,): 5
    }
    assert result.card_count == 5
    decoded = AnonymousDeckComponentProfile.model_validate_json(
        result.model_dump_json()
    )
    assert decoded == result


def test_duplicate_isomorphic_rows_accumulate_without_canonicalization() -> None:
    # Component sizes are relabelling invariants, so two non-canonical but
    # isomorphic representatives must merge under the same component tuple
    # without any permutation canonicalization.
    first = SimpleUndirectedGraph(
        vertices=("v00", "v01", "v02", "v03"),
        edges=(("v00", "v01"), ("v01", "v02"), ("v02", "v03")),
    )
    second = SimpleUndirectedGraph(
        vertices=("v00", "v01", "v02", "v03"),
        edges=(("v00", "v01"), ("v01", "v03"), ("v02", "v03")),
    )
    deck = AnonymousGraphCardMultiset.model_construct(
        card_order=4,
        classes=(
            AnonymousGraphCardClass.model_construct(
                representative=first, multiplicity=2
            ),
            AnonymousGraphCardClass.model_construct(
                representative=second, multiplicity=3
            ),
        ),
    )
    result = card_component_profile(
        AnonymousDeckComponentProfileRequest.model_construct(deck=deck)
    )
    assert {item.component_orders: item.multiplicity for item in result.profiles} == {
        (4,): 5
    }
    assert result.card_count == 5


@pytest.mark.parametrize(
    "forged_graph",
    (
        SimpleUndirectedGraph.model_construct(),
        SimpleUndirectedGraph.model_construct(vertices=("v00", "v01")),
    ),
)
def test_schema_bypassed_representative_reports_domain_error(
    forged_graph: SimpleUndirectedGraph,
) -> None:
    # Schema-bypassed nested values must yield the declared domain diagnostic,
    # not a raw AttributeError from missing graph fields.
    deck = AnonymousGraphCardMultiset.model_construct(
        card_order=2,
        classes=(
            AnonymousGraphCardClass.model_construct(
                representative=forged_graph, multiplicity=1
            ),
        ),
    )
    with pytest.raises(OperationDomainValidationError):
        card_component_profile(
            AnonymousDeckComponentProfileRequest.model_construct(deck=deck)
        )


def test_connectivity_work_is_admitted_before_graph_traversal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = SimpleUndirectedGraph(
        vertices=("v00", "v01", "v02"), edges=(("v00", "v01"),)
    )
    deck = AnonymousGraphCardMultiset.model_construct(
        card_order=3,
        classes=(
            AnonymousGraphCardClass.model_construct(
                representative=graph, multiplicity=1
            ),
        ),
    )
    monkeypatch.setattr(
        "jacobian.math.graphs.decks.card_component_profile.operations"
        ".MAX_CARD_COMPONENT_PROFILE_WORK",
        0,
    )
    calls = 0

    def unexpected(*_args: object) -> None:
        nonlocal calls
        calls += 1
        raise AssertionError("component traversal preceded admission")

    monkeypatch.setattr(
        "jacobian.math.graphs.decks.card_component_profile.operations._component_orders",
        unexpected,
    )
    with pytest.raises(OperationResourceAdmissionError):
        card_component_profile(
            AnonymousDeckComponentProfileRequest.model_construct(deck=deck)
        )
    assert calls == 0
