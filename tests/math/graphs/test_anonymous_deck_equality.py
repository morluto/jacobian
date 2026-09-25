"""Exact equality of anonymous graph-card multisets."""

from itertools import combinations, permutations

import pytest
from pydantic import ValidationError

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationMatchRequest,
)
from jacobian.dispatch import invoke_operation
from jacobian.math.graphs.decks import _models as deck_models
from jacobian.math.graphs.decks._models import (
    AnonymousGraphCardClass,
    AnonymousGraphCardMultiset,
    AnonymousGraphCardMultisetEqualityRequest,
    AnonymousGraphCardMultisetRequest,
)
from jacobian.math.graphs.decks.operations import (
    anonymous_graph_card_multiset,
    anonymous_graph_card_multiset_equal,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _all_graphs(order: int):
    vertices = tuple(f"v{i}" for i in range(order))
    possible_edges = tuple(combinations(vertices, 2))
    for mask in range(1 << len(possible_edges)):
        yield SimpleUndirectedGraph(
            vertices=vertices,
            edges=tuple(
                edge for bit, edge in enumerate(possible_edges) if mask & (1 << bit)
            ),
        )


def _isomorphic(left: SimpleUndirectedGraph, right: SimpleUndirectedGraph) -> bool:
    """Independent tiny-graph oracle by checking every vertex bijection."""
    if len(left.vertices) != len(right.vertices):
        return False
    right_edges = {frozenset(edge) for edge in right.edges}
    for image in permutations(right.vertices):
        mapping = dict(zip(left.vertices, image, strict=True))
        if {frozenset((mapping[u], mapping[v])) for u, v in left.edges} == right_edges:
            return True
    return False


def _multiset(cards: tuple[SimpleUndirectedGraph, ...], order: int):
    return anonymous_graph_card_multiset(
        AnonymousGraphCardMultisetRequest(card_order=order, cards=cards)
    )


def test_exhaustive_order_three_single_card_equality_matches_permutation_oracle() -> (
    None
):
    graphs = tuple(_all_graphs(3))
    for left in graphs:
        for right in graphs:
            request = AnonymousGraphCardMultisetEqualityRequest(
                left=_multiset((left,), 3), right=_multiset((right,), 3)
            )
            result = anonymous_graph_card_multiset_equal(request)
            assert result.equal is _isomorphic(left, right)


def test_relabeling_row_order_multiplicity_and_card_order_semantics() -> None:
    path = SimpleUndirectedGraph(
        vertices=("a", "b", "c"), edges=(("a", "b"), ("b", "c"))
    )
    relabelled_path = SimpleUndirectedGraph(
        vertices=("z", "x", "y"), edges=(("x", "y"), ("y", "z"))
    )
    empty = SimpleUndirectedGraph(vertices=("a", "b", "c"), edges=())
    same = AnonymousGraphCardMultisetEqualityRequest(
        left=_multiset((path, empty), 3),
        right=_multiset((empty, relabelled_path), 3),
    )
    assert anonymous_graph_card_multiset_equal(same).equal
    round_trip = AnonymousGraphCardMultisetEqualityRequest.model_validate_json(
        same.model_dump_json()
    )
    assert anonymous_graph_card_multiset_equal(round_trip).equal

    changed_multiplicity = AnonymousGraphCardMultisetEqualityRequest(
        left=_multiset((path, empty), 3), right=_multiset((path, path), 3)
    )
    assert not anonymous_graph_card_multiset_equal(changed_multiplicity).equal

    changed_order = AnonymousGraphCardMultisetEqualityRequest(
        left=_multiset((), 2), right=_multiset((), 3)
    )
    assert not anonymous_graph_card_multiset_equal(changed_order).equal


def test_catalog_discovers_and_runs_anonymous_multiset_equality() -> None:
    catalog = Catalog.open()
    match = catalog.match(
        OperationMatchRequest(need="compare equality of anonymous graph card multisets")
    )
    assert match.matches[0].operation_id == "graph.deck.anonymous_multiset.equal.check"
    operation = catalog.operation(match.matches[0].operation_id)
    assert operation is not None
    output = invoke_operation(
        operation.operation_id,
        {
            "left": {"card_order": 1, "classes": []},
            "right": {"card_order": 1, "classes": []},
        },
        catalog,
    )
    assert output.output["equal"]
    example_result = invoke_operation(
        operation.operation_id, operation.examples[0].input, catalog
    )
    assert example_result.output["equal"] is False


def test_combined_canonical_validation_work_is_admitted_once_for_both_sides() -> None:
    cards = (
        SimpleUndirectedGraph(vertices=tuple("abcdefg"), edges=()),
        SimpleUndirectedGraph(vertices=tuple("abcdefg"), edges=(("a", "b"),)),
        SimpleUndirectedGraph(
            vertices=tuple("abcdefg"), edges=(("a", "b"), ("c", "d"))
        ),
        SimpleUndirectedGraph(
            vertices=tuple("abcdefg"), edges=(("a", "b"), ("b", "c"))
        ),
    )
    four_classes = _multiset(cards, 7)
    accepted = AnonymousGraphCardMultisetEqualityRequest.model_validate(
        {"left": four_classes.model_dump(), "right": four_classes.model_dump()}
    )
    assert anonymous_graph_card_multiset_equal(accepted).equal

    five_classes = {
        "card_order": 7,
        "classes": [
            *four_classes.model_dump()["classes"],
            {
                "representative": {
                    "vertices": list(four_classes.classes[0].representative.vertices),
                    "edges": [["v00", "v01"], ["v02", "v03"]],
                },
                "multiplicity": 1,
            },
        ],
    }
    with pytest.raises(ValidationError, match="combined canonical validation"):
        AnonymousGraphCardMultisetEqualityRequest.model_validate(
            {"left": five_classes, "right": four_classes.model_dump()}
        )


def test_native_and_catalog_paths_do_not_replay_canonical_validation(
    monkeypatch,
) -> None:
    from jacobian.math.graphs.decks.operations import (
        anonymous_graph_card_multiset_equal,
    )

    raw = {
        "card_order": 1,
        "classes": [
            {
                "representative": {"vertices": ["v00"], "edges": []},
                "multiplicity": "1",
            }
        ],
    }
    payload = {"left": raw, "right": raw}
    original = deck_models._canonical_card_edges
    canonical_checks = 0

    def count_canonical_checks(vertices, edges):
        nonlocal canonical_checks
        canonical_checks += 1
        return original(vertices, edges)

    monkeypatch.setattr(deck_models, "_canonical_card_edges", count_canonical_checks)
    native_payload = {
        side: {
            **raw,
            "classes": [{**raw["classes"][0], "multiplicity": 1}],
        }
        for side in ("left", "right")
    }
    request = AnonymousGraphCardMultisetEqualityRequest.model_validate(native_payload)
    assert canonical_checks == 2
    assert anonymous_graph_card_multiset_equal(request).equal
    assert canonical_checks == 2

    canonical_checks = 0
    catalog = Catalog.open()
    operation = catalog.operation("graph.deck.anonymous_multiset.equal.check")
    assert operation is not None
    assert invoke_operation(operation.operation_id, payload, catalog).output["equal"]
    assert canonical_checks == 2


def test_typed_composition_is_preflighted_revalidated_once_and_roundtrips(
    monkeypatch,
) -> None:
    graph = SimpleUndirectedGraph(vertices=("a",), edges=())
    left = _multiset((graph,), 1)
    right = _multiset((graph,), 1)
    assert not hasattr(left, "_canonical_snapshot")

    original = deck_models._canonical_card_edges
    canonical_checks = 0

    def count_canonical_checks(vertices, edges):
        nonlocal canonical_checks
        canonical_checks += 1
        return original(vertices, edges)

    monkeypatch.setattr(deck_models, "_canonical_card_edges", count_canonical_checks)
    request = AnonymousGraphCardMultisetEqualityRequest(left=left, right=right)
    assert canonical_checks == 2
    assert anonymous_graph_card_multiset_equal(request).equal
    assert canonical_checks == 2

    round_trip = AnonymousGraphCardMultisetEqualityRequest.model_validate_json(
        request.model_dump_json()
    )
    assert canonical_checks == 4
    assert anonymous_graph_card_multiset_equal(round_trip).equal
    assert canonical_checks == 4


def test_native_path_rejects_unvalidated_or_modified_request_carriers() -> None:
    from jacobian.math.graphs.decks.operations import (
        anonymous_graph_card_multiset_equal,
    )

    unchecked_multiset = AnonymousGraphCardMultiset.model_construct(
        card_order=10,
        classes=(
            AnonymousGraphCardClass.model_construct(
                representative=SimpleUndirectedGraph.model_construct(
                    vertices=tuple(f"v{i:02d}" for i in range(10)), edges=()
                ),
                multiplicity=1,
            ),
        ),
    )
    with pytest.raises(ValidationError, match="combined canonical validation"):
        AnonymousGraphCardMultisetEqualityRequest(
            left=unchecked_multiset, right=unchecked_multiset
        )

    noncanonical_multiset = AnonymousGraphCardMultiset.model_construct(
        card_order=3,
        classes=(
            AnonymousGraphCardClass.model_construct(
                representative=SimpleUndirectedGraph.model_construct(
                    vertices=("v00", "v01", "v02"), edges=(("v00", "v01"),)
                ),
                multiplicity=1,
            ),
        ),
    )
    with pytest.raises(ValidationError, match="minimal under all vertex permutations"):
        AnonymousGraphCardMultisetEqualityRequest(
            left=noncanonical_multiset, right=noncanonical_multiset
        )

    forged = AnonymousGraphCardMultisetEqualityRequest.model_construct(
        left=unchecked_multiset, right=unchecked_multiset
    )
    with pytest.raises(OperationDomainValidationError, match="combined admission"):
        anonymous_graph_card_multiset_equal(forged)

    valid = AnonymousGraphCardMultisetEqualityRequest(
        left=_multiset((), 0), right=_multiset((), 0)
    )
    modified = valid.model_copy(update={"left": unchecked_multiset})
    with pytest.raises(OperationDomainValidationError, match="combined admission"):
        anonymous_graph_card_multiset_equal(modified)
