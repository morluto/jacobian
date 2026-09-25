from __future__ import annotations

import json
from math import comb, factorial

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.dispatch import OperationRequestValidationError, invoke_operation
from jacobian.math.graphs.decks import (
    AnonymousCardDegreeProfile,
    AnonymousCardDegreeProfileRequest,
    AnonymousGraphCardClass,
    AnonymousGraphCardMultiset,
    AnonymousGraphCardMultisetRequest,
    anonymous_card_degree_profile,
    anonymous_graph_card_multiset,
)
from jacobian.math.graphs.decks import _models as deck_models
from jacobian.math.graphs.decks import operations as deck_operations
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _cycle6(prefix: str) -> SimpleUndirectedGraph:
    vertices = tuple(f"{prefix}{i}" for i in range(6))
    return SimpleUndirectedGraph(
        vertices=vertices,
        edges=tuple(
            tuple(sorted((vertices[i], vertices[(i + 1) % 6]))) for i in range(6)
        ),
    )


def _two_triangles(prefix: str) -> SimpleUndirectedGraph:
    vertices = tuple(f"{prefix}{i}" for i in range(6))
    return SimpleUndirectedGraph(
        vertices=vertices,
        edges=tuple(
            (vertices[i], vertices[j])
            for start in (0, 3)
            for i, j in ((start, start + 1), (start, start + 2), (start + 1, start + 2))
        ),
    )


def _path6(prefix: str) -> SimpleUndirectedGraph:
    vertices = tuple(f"{prefix}{i}" for i in range(6))
    return SimpleUndirectedGraph(
        vertices=vertices,
        edges=tuple((vertices[i], vertices[i + 1]) for i in range(5)),
    )


def _profile(cards: tuple[SimpleUndirectedGraph, ...], order: int):
    multiset = anonymous_graph_card_multiset(
        AnonymousGraphCardMultisetRequest(card_order=order, cards=cards)
    )
    return multiset, anonymous_card_degree_profile(multiset)


def test_cycles_paths_and_nonisomorphic_regular_cards_share_only_their_degree_invariant() -> (
    None
):
    multiset, profile = _profile(
        (
            _cycle6("c"),
            _cycle6("x"),
            _two_triangles("t"),
            _two_triangles("u"),
            _two_triangles("w"),
            _path6("p"),
            _path6("q"),
            _path6("r"),
            _path6("s"),
        ),
        6,
    )
    assert len(multiset.classes) == 3
    assert len(profile.degree_multisets) == 2
    assert profile.card_order == 6
    assert profile.total_card_multiplicity == 9
    assert tuple(row.degrees.degrees for row in profile.degree_multisets) == (
        (2, 2, 2, 2, 1, 1),
        (2, 2, 2, 2, 2, 2),
    )
    assert tuple(row.multiplicity for row in profile.degree_multisets) == (4, 5)


def test_profile_is_relabel_invariant_and_sensitive_to_card_multiplicity() -> None:
    _, first = _profile((_cycle6("a"), _two_triangles("b")), 6)
    _, relabelled = _profile((_cycle6("u"), _two_triangles("v")), 6)
    _, doubled = _profile((_cycle6("m"), _cycle6("n"), _two_triangles("o")), 6)
    assert first == relabelled
    assert len(first.degree_multisets) == len(doubled.degree_multisets) == 1
    assert first.degree_multisets[0].multiplicity == 2
    assert doubled.degree_multisets[0].multiplicity == 3


def test_empty_profile_retains_card_order_and_round_trips() -> None:
    multiset = anonymous_graph_card_multiset(
        AnonymousGraphCardMultisetRequest(card_order=7, cards=())
    )
    result = anonymous_card_degree_profile(multiset)
    assert result.card_order == 7
    assert result.total_card_multiplicity == 0
    assert result.degree_multisets == ()
    assert (
        AnonymousCardDegreeProfile.model_validate_json(result.model_dump_json())
        == result
    )


def test_native_operation_rejects_model_construct_noncanonical_card() -> None:
    representative = SimpleUndirectedGraph.model_construct(
        vertices=("v00", "v01", "v02"), edges=(("v00", "v01"),)
    )
    card_class = AnonymousGraphCardClass.model_construct(
        representative=representative, multiplicity=1
    )
    multiset = AnonymousGraphCardMultiset.model_construct(
        card_order=3, classes=(card_class,)
    )
    request = AnonymousCardDegreeProfileRequest.model_construct(multiset=multiset)
    with pytest.raises(OperationDomainValidationError, match="permutation-minimal"):
        anonymous_card_degree_profile(request.multiset)


def test_native_admission_charges_one_canonicalization_per_card_class(
    monkeypatch,
) -> None:
    _, valid_profile = _profile((_cycle6("a"), _two_triangles("b")), 6)
    multiset, _ = _profile((_cycle6("c"), _two_triangles("d")), 6)
    request = AnonymousCardDegreeProfileRequest.model_construct(multiset=multiset)
    original = deck_operations._canonical_card_edges
    calls = 0

    def counted(vertices, edges):
        nonlocal calls
        calls += 1
        return original(vertices, edges)

    monkeypatch.setattr(deck_operations, "_canonical_card_edges", counted)
    result = anonymous_card_degree_profile(request.multiset)
    assert result == valid_profile
    assert calls == len(multiset.classes)


def test_profile_work_cells_and_output_are_admitted_at_exact_boundaries(
    monkeypatch,
) -> None:
    multiset = anonymous_graph_card_multiset(
        AnonymousGraphCardMultisetRequest(
            card_order=4,
            cards=(
                SimpleUndirectedGraph(
                    vertices=("a", "b", "c", "d"),
                    edges=(("a", "b"), ("b", "c")),
                ),
            ),
        )
    )
    request = AnonymousCardDegreeProfileRequest(multiset=multiset)
    canonical_work = factorial(4) * (4 + 2 * comb(4, 2))
    profile_work = 4 * 4 + 3 * 4 + 4 * comb(4, 2) + 4 + 4
    exact_work = canonical_work + profile_work
    monkeypatch.setattr(deck_operations, "MAX_ANONYMOUS_CARD_PROFILE_WORK", exact_work)
    assert anonymous_card_degree_profile(request.multiset).degree_multisets
    monkeypatch.setattr(
        deck_operations, "MAX_ANONYMOUS_CARD_PROFILE_WORK", exact_work - 1
    )
    with pytest.raises(OperationResourceAdmissionError, match="shared work bound"):
        anonymous_card_degree_profile(request.multiset)

    monkeypatch.setattr(deck_operations, "MAX_ANONYMOUS_CARD_PROFILE_WORK", 2_000_000)
    monkeypatch.setattr(deck_operations, "MAX_ANONYMOUS_CARD_PROFILE_CELLS", 4)
    assert anonymous_card_degree_profile(request.multiset).degree_multisets
    monkeypatch.setattr(deck_operations, "MAX_ANONYMOUS_CARD_PROFILE_CELLS", 3)
    with pytest.raises(OperationResourceAdmissionError, match="cells"):
        anonymous_card_degree_profile(request.multiset)

    monkeypatch.setattr(deck_operations, "MAX_ANONYMOUS_CARD_PROFILE_CELLS", 200_000)
    output_bound = 128 + 64 + 16 * 4
    monkeypatch.setattr(
        deck_operations, "MAX_ANONYMOUS_CARD_PROFILE_RESULT_BYTES", output_bound
    )
    assert anonymous_card_degree_profile(request.multiset).degree_multisets
    monkeypatch.setattr(
        deck_operations, "MAX_ANONYMOUS_CARD_PROFILE_RESULT_BYTES", output_bound - 1
    )
    with pytest.raises(OperationResourceAdmissionError, match="byte bound"):
        anonymous_card_degree_profile(request.multiset)


def test_catalog_round_trip_canonicalizes_nested_multiset_only_once(
    monkeypatch,
) -> None:
    operation = Catalog.open().operation("graph.deck.card_invariant_profile.compute")
    assert operation is not None
    original = deck_models._canonical_card_edges
    calls = 0

    def counted(vertices, edges):
        nonlocal calls
        calls += 1
        return original(vertices, edges)

    monkeypatch.setattr(deck_models, "_canonical_card_edges", counted)
    exact_total_work = deck_models._anonymous_profile_resource_estimates(3, 2)[1]
    monkeypatch.setattr(
        deck_models, "MAX_ANONYMOUS_CARD_PROFILE_WORK", exact_total_work
    )
    invocation = invoke_operation(
        operation.operation_id, operation.examples[0].input, Catalog.open()
    )
    assert invocation.output
    assert calls == 2
    decoded = operation.result_type.model_validate_json(json.dumps(invocation.output))
    assert decoded.total_card_multiplicity == 3


def test_catalog_rejects_combined_bound_before_nested_canonicalization(
    monkeypatch,
) -> None:
    operation = Catalog.open().operation("graph.deck.card_invariant_profile.compute")
    assert operation is not None
    payload = operation.examples[0].input
    original = deck_models._canonical_card_edges
    calls = 0

    def counted(vertices, edges):
        nonlocal calls
        calls += 1
        return original(vertices, edges)

    monkeypatch.setattr(deck_models, "_canonical_card_edges", counted)
    monkeypatch.setattr(deck_models, "MAX_ANONYMOUS_CARD_PROFILE_WORK", 1)
    with pytest.raises(OperationRequestValidationError) as error:
        invoke_operation(operation.operation_id, payload, Catalog.open())
    assert "shared work bound" in str(error.value.cause)
    assert calls == 0
