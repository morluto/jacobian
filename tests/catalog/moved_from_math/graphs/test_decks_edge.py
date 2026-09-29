"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/graphs/test_decks_edge.py``. The preamble below is carried over verbatim so the moved
tests resolve every name they use.
"""

from __future__ import annotations

from itertools import permutations

from jacobian.math.graphs.decks._models import (
    UnlabelledDeck,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _triangle() -> SimpleUndirectedGraph:
    return SimpleUndirectedGraph(
        vertices=("a", "b", "c"),
        edges=(("a", "b"), ("a", "c"), ("b", "c")),
    )


def _oracle_signature(graph: SimpleUndirectedGraph) -> tuple[int, ...]:
    """Independent adjacency-matrix permutation oracle for tiny graphs."""
    labels = graph.vertices
    adjacency = {frozenset(edge) for edge in graph.edges}
    candidates = []
    for order in permutations(labels):
        candidates.append(
            tuple(
                int(frozenset((order[i], order[j])) in adjacency)
                for i in range(len(order))
                for j in range(i + 1, len(order))
            )
        )
    return min(candidates, default=())


def test_edge_deck_quotient_has_exactly_one_published_operation() -> None:
    """One postcondition, one operation: no competing edge-deck quotient IDs.

    The guard keys on the *result* carrier, not on the request input. Several
    published operations legitimately consume an ``EdgeDeletionFamily`` -- the
    quotient, and the classification profile that returns an explicit
    permutation witness per card -- so matching on the input alone would flag
    distinct postconditions as competing. Keying on the result is what actually
    asserts "one quotient, one operation", and it keeps the guard strict: a
    second operation returning ``UnlabelledDeck`` is still rejected.
    """
    from jacobian.catalog.builtins import BUILTIN_TOOLS

    quotients = sorted(
        tool.operation_id
        for tool in BUILTIN_TOOLS
        if tool.result_type is UnlabelledDeck
    )
    assert quotients == ["graph.deck.unlabelled.compute"]
