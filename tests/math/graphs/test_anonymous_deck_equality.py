import json
from itertools import permutations

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.graphs.decks._models import (
    AnonymousGraphCardClass,
    AnonymousGraphCardMultiset,
)
from jacobian.math.graphs.decks.anonymous_equality._tools import TOOLS
from jacobian.math.graphs.decks.anonymous_equality.operations import (
    anonymous_deck_equality,
)
from jacobian.math.graphs.decks.operations import anonymous_graph_card_multiset
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _multiset(*cards: SimpleUndirectedGraph) -> AnonymousGraphCardMultiset:
    order = len(cards[0].vertices) if cards else 0
    return anonymous_graph_card_multiset(order, cards)


def _independent_orbit_edges(
    graph: SimpleUndirectedGraph,
) -> tuple[tuple[str, ...], ...]:
    labels = graph.vertices
    rows: list[tuple[tuple[str, ...], ...]] = []
    for order in permutations(labels):
        renamed = tuple(
            sorted(
                tuple(sorted((f"v{order.index(a):02d}", f"v{order.index(b):02d}")))
                for a, b in graph.edges
            )
        )
        rows.append(renamed)
    return min(rows, default=())


def test_independent_card_relabelling_preserves_multiset_equality() -> None:
    path = SimpleUndirectedGraph(
        vertices=("a", "b", "c", "d"),
        edges=(("a", "b"), ("b", "c"), ("c", "d")),
    )
    relabelled_path = SimpleUndirectedGraph(
        vertices=("w", "x", "y", "z"),
        edges=(("w", "y"), ("x", "y"), ("x", "z")),
    )
    triangle = SimpleUndirectedGraph(
        vertices=("p", "q", "r", "s"),
        edges=(("p", "q"), ("p", "r"), ("q", "r")),
    )
    left, right = _multiset(path, triangle), _multiset(relabelled_path, triangle)
    assert _independent_orbit_edges(path) == _independent_orbit_edges(relabelled_path)
    assert anonymous_deck_equality(left, right).equal


def test_equality_checks_noncanonical_wire_representatives_by_isomorphism() -> None:
    canonical = _multiset(
        SimpleUndirectedGraph(
            vertices=("a", "b", "c"),
            edges=(("a", "b"), ("b", "c")),
        )
    )
    relabelled = SimpleUndirectedGraph(
        vertices=("v00", "v01", "v02"),
        edges=(("v00", "v01"), ("v00", "v02")),
    )
    assert canonical.classes[0].representative.edges != relabelled.edges
    noncanonical_wire_value = AnonymousGraphCardMultiset.model_construct(
        card_order=3,
        classes=(
            AnonymousGraphCardClass.model_construct(
                representative=relabelled, multiplicity=1
            ),
        ),
    )
    assert anonymous_deck_equality(canonical, noncanonical_wire_value).equal


def test_equal_underlying_sets_with_different_multiplicity_are_not_equal() -> None:
    vertices = ("a", "b", "c")
    edge = SimpleUndirectedGraph(vertices=vertices, edges=(("a", "b"),))
    empty = SimpleUndirectedGraph(vertices=vertices, edges=())
    left, right = _multiset(edge, edge, empty), _multiset(edge, empty, empty)
    assert not anonymous_deck_equality(left, right).equal


def test_empty_multisets_retain_and_compare_their_card_order() -> None:
    zero = _multiset()
    order_two = AnonymousGraphCardMultiset(card_order=2, classes=())
    assert anonymous_deck_equality(zero, zero).equal
    assert not anonymous_deck_equality(zero, order_two).equal


def test_forged_catalog_request_missing_right_is_a_domain_error() -> None:
    tool = next(
        item
        for item in TOOLS
        if item.operation_id == "graph.deck.anonymous.equal.decide"
    )
    with pytest.raises(OperationDomainValidationError) as exc_info:
        tool.run(tool.request_type.model_construct(left=_multiset()))
    assert exc_info.value.errors()[0]["type"] == "graph_deck.equality_carrier"


def test_published_catalog_example_executes() -> None:
    tool = next(
        item
        for item in TOOLS
        if item.operation_id == "graph.deck.anonymous.equal.decide"
    )
    request = tool.request_type.model_validate_json(
        json.dumps(tool.examples[0].input), strict=True
    )
    assert tool.run(request).equal


@pytest.mark.parametrize("missing_field", ["vertices", "edges"])
def test_forged_fieldless_representatives_are_rejected_at_admission(
    missing_field: str,
) -> None:
    valid = _multiset(SimpleUndirectedGraph(vertices=("v00",), edges=()))
    if missing_field == "vertices":
        fieldless = SimpleUndirectedGraph.model_construct(edges=())
    else:
        fieldless = SimpleUndirectedGraph.model_construct(vertices=("v00",))
    forged_class = AnonymousGraphCardClass.model_construct(
        representative=fieldless,
        multiplicity=1,
    )
    forged = AnonymousGraphCardMultiset.model_construct(
        card_order=1, classes=(forged_class,)
    )
    with pytest.raises(OperationDomainValidationError, match="bounded representative"):
        anonymous_deck_equality(forged, valid)


def test_admits_aggregate_work_before_canonicalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    one_class = AnonymousGraphCardClass(
        representative=SimpleUndirectedGraph(vertices=("v00",), edges=()),
        multiplicity=1,
    )
    many = AnonymousGraphCardMultiset.model_construct(
        card_order=1, classes=(one_class,) * 10_000
    )
    # Each side fits individually, but the pair exceeds the shared limit.
    monkeypatch.setattr(
        "jacobian.math.graphs.decks.anonymous_equality.operations._anonymous_canonicalization_work",
        lambda _order, count: count * 128,
    )
    calls = 0

    def should_not_canonicalize(*_args: object) -> None:
        nonlocal calls
        calls += 1
        raise AssertionError("canonicalization must follow aggregate admission")

    monkeypatch.setattr(
        "jacobian.math.graphs.decks.anonymous_equality.operations._canonical_card_edges",
        should_not_canonicalize,
    )
    with pytest.raises(OperationResourceAdmissionError):
        anonymous_deck_equality(many, many)
    assert calls == 0
