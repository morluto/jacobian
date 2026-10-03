"""Catalog projection check for the card-component deck profile."""

from jacobian.math.graphs.decks._models import (
    AnonymousGraphCardMultiset,
)
from jacobian.math.graphs.decks.card_component_profile.operations import (
    card_component_profile,
)
from jacobian.math.graphs.decks.operations import anonymous_graph_card_multiset
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _deck(cards: tuple[SimpleUndirectedGraph, ...]) -> AnonymousGraphCardMultiset:
    return anonymous_graph_card_multiset(4, cards)


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


def test_card_component_profile_is_native_only_projection() -> None:
    from jacobian.catalog.builtins import BUILTIN_TOOLS
    from jacobian.math.graphs.decks.card_component_profile import (
        card_component_profile as public_profile,
    )

    assert public_profile is card_component_profile
    assert all(
        tool.operation_id != "graph.deck.card_component_profile.compute"
        for tool in BUILTIN_TOOLS
    )
    assert card_component_profile(_deck(())).card_count == 0
