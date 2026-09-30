"""Regressions for the deck profile decode boundary and family carrier.

Result decoding must not replay the producer's canonicalization search, and the
normalizer must bound attacker-controlled rows before copying them.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.decks import _models as models_module
from jacobian.math.graphs.decks._models import (
    EdgeDeckIsomorphismProfile,
    VertexDeckIsomorphismProfile,
    VertexDeletionFamily,
)
from jacobian.math.graphs.decks.operations import (
    edge_deck_isomorphism_profile,
    edge_deletion_family,
    unlabelled_vertex_deck,
    vertex_deck_isomorphism_profile,
    vertex_deletion_family,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _source(order: int = 4) -> SimpleUndirectedGraph:
    vertices = tuple(f"v{index:02d}" for index in range(order))
    edges = tuple(
        (vertices[left], vertices[right])
        for left in range(order)
        for right in range(left + 1, order)
    )
    return SimpleUndirectedGraph(vertices=vertices, edges=edges)


def _profile(order: int = 4) -> VertexDeckIsomorphismProfile:
    return vertex_deck_isomorphism_profile(vertex_deletion_family(_source(order)))


# --- result decoding does not replay the canonicalization search -----------


@pytest.mark.parametrize("edge_deck", [False, True])
def test_result_decode_does_not_recanonicalize(
    monkeypatch: pytest.MonkeyPatch,
    edge_deck: bool,
) -> None:
    """Decoding an admitted result must not enumerate vertex permutations."""
    original = (
        edge_deck_isomorphism_profile(edge_deletion_family(_source()))
        if edge_deck
        else _profile()
    )
    calls = 0
    real = models_module._canonical_card_edges

    def counting(
        vertices: tuple[str, ...], edges: tuple[tuple[str, str], ...]
    ) -> tuple[tuple[str, str], ...]:
        nonlocal calls
        calls += 1
        return real(vertices, edges)

    monkeypatch.setattr(models_module, "_canonical_card_edges", counting)
    assert type(original).model_validate(original.model_dump()) == original
    assert type(original).model_validate_json(original.model_dump_json()) == original
    assert calls == 0


def test_canonicalization_is_still_used_by_the_producer() -> None:
    """K1,3 gives noncanonical cards with independently known normal forms."""
    source = SimpleUndirectedGraph(
        vertices=("a", "b", "c", "d"),
        edges=(("a", "b"), ("a", "c"), ("a", "d")),
    )
    vertex_profile = vertex_deck_isomorphism_profile(vertex_deletion_family(source))
    # Deleting the center leaves three isolated vertices. Deleting a leaf
    # leaves P3, whose least adjacency bit string is 011, with center v02.
    assert tuple(
        (item.representative.edges, item.multiplicity)
        for item in vertex_profile.classes
    ) == (((), 1), ((("v00", "v02"), ("v01", "v02")), 3))
    assert vertex_profile.class_indices == (0, 1, 1, 1)
    assert (
        VertexDeckIsomorphismProfile.model_validate_json(
            vertex_profile.model_dump_json()
        )
        == vertex_profile
    )

    edge_profile = edge_deck_isomorphism_profile(edge_deletion_family(source))
    # Every edge deletion leaves P3 plus one isolated vertex: 000011.
    assert tuple(
        (item.representative.edges, item.multiplicity) for item in edge_profile.classes
    ) == (((("v01", "v03"), ("v02", "v03")), 3),)
    assert edge_profile.class_indices == (0, 0, 0)
    assert (
        EdgeDeckIsomorphismProfile.model_validate_json(edge_profile.model_dump_json())
        == edge_profile
    )


# --- oversized rows are bounded before the normalizer copies them ----------


def test_oversized_vertex_map_row_is_bounded_before_copying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refusal must precede the normalizer that copies the row."""
    payload = _profile().model_dump()
    payload["vertex_maps"] = [[0] * 4 for _ in range(3)]
    payload["vertex_maps"][0] = [0] * 500_000
    reached: list[bool] = []
    real = models_module._normalize_vertex_iso_profile_result

    def spy(value: Any) -> Any:
        reached.append(True)
        return real(value)

    monkeypatch.setattr(models_module, "_normalize_vertex_iso_profile_result", spy)
    with pytest.raises(ValueError):
        VertexDeckIsomorphismProfile.model_validate(payload)
    assert not reached, "the oversized row was copied before being refused"


def test_oversized_class_index_rows_are_bounded_before_copying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _profile().model_dump()
    payload["class_indices"] = [0] * 500_000
    reached: list[bool] = []
    real = models_module._normalize_vertex_iso_profile_result

    def spy(value: Any) -> Any:
        reached.append(True)
        return real(value)

    monkeypatch.setattr(models_module, "_normalize_vertex_iso_profile_result", spy)
    with pytest.raises(ValueError):
        VertexDeckIsomorphismProfile.model_validate(payload)
    assert not reached, "the oversized index rows were copied before being refused"


def test_oversized_class_card_indices_are_bounded() -> None:
    payload = _profile().model_dump()
    payload["classes"][0]["card_indices"] = [0] * 500_000
    with pytest.raises(ValueError):
        VertexDeckIsomorphismProfile.model_validate(payload)


def test_oversized_class_representative_is_bounded() -> None:
    payload = _profile().model_dump()
    payload["classes"][0]["representative"]["edges"] = [["v00", "v01"]] * 5_000
    with pytest.raises(ValueError):
        VertexDeckIsomorphismProfile.model_validate(payload)


@pytest.mark.parametrize("wire", [False, True])
@pytest.mark.parametrize("field", ["vertices", "edge_endpoints"])
def test_representative_axes_are_bounded_before_normalization(
    monkeypatch: pytest.MonkeyPatch, wire: bool, field: str
) -> None:
    payload = _profile().model_dump(mode="json")
    representative = payload["classes"][0]["representative"]
    if field == "vertices":
        representative["vertices"] = ["v00"] * 50_000
    else:
        representative["edges"] = [["v00"] * 50_000]
    encoded = json.dumps(payload)
    reached = False
    real = models_module._normalize_vertex_iso_profile_result

    def spy(value: Any) -> Any:
        nonlocal reached
        reached = True
        return real(value)

    monkeypatch.setattr(models_module, "_normalize_vertex_iso_profile_result", spy)
    with pytest.raises(ValueError):
        if wire:
            VertexDeckIsomorphismProfile.model_validate_json(encoded)
        else:
            VertexDeckIsomorphismProfile.model_validate(payload)
    assert not reached


# --- a forged family carrier is refused, not dereferenced -----------------


def test_forged_family_carrier_is_refused() -> None:
    """A family with no source must raise the operation's own error."""
    forged = VertexDeletionFamily.model_construct()
    with pytest.raises(OperationDomainValidationError) as error:
        unlabelled_vertex_deck(forged)
    assert str(error.value.errors()[0]["type"]) == ("graph_deck.vertex_family_carrier")


def test_forged_family_with_non_tuple_cards_is_refused() -> None:
    family = vertex_deletion_family(_source())
    forged = family.model_copy(update={"cards": list(family.cards)})
    with pytest.raises(OperationDomainValidationError):
        unlabelled_vertex_deck(forged)


# --- negative controls -----------------------------------------------------


@pytest.mark.parametrize("order", [0, 1, 8])
def test_profile_round_trips_unchanged(order: int) -> None:
    original = _profile(order)
    again = VertexDeckIsomorphismProfile.model_validate(original.model_dump())
    assert again == original
    assert (
        VertexDeckIsomorphismProfile.model_validate_json(
            original.model_dump_json(), strict=True
        )
        == original
    )
    assert sum(item.multiplicity for item in original.classes) == order
    if order:
        assert len(original.classes) == 1
        assert original.classes[0].representative == _source(order - 1)
    else:
        assert original.classes == ()


def test_valid_family_is_still_quotiented() -> None:
    deck = unlabelled_vertex_deck(vertex_deletion_family(_source()))
    assert deck.card_count == 4
    assert len(deck.classes) >= 1


def test_a_forged_vertex_map_is_still_refused() -> None:
    """The structural bijection checks are retained after the preflight change."""
    payload = _profile().model_dump()
    maps = [list(row) for row in payload["vertex_maps"]]
    # A repeated entry is not a bijection, unlike a reordering of a
    # symmetric complete graph, which is a legitimate alternative map.
    maps[0] = [maps[0][0]] * len(maps[0])
    payload["vertex_maps"] = [tuple(row) for row in maps]
    with pytest.raises(ValueError):
        VertexDeckIsomorphismProfile.model_validate(payload)


@pytest.mark.parametrize("wire", [False, True])
def test_authored_vertex_partition_does_not_authenticate_its_quotient(
    wire: bool,
) -> None:
    """A split P3 class is structural data, not evidence of distinct classes."""
    source = SimpleUndirectedGraph(
        vertices=("a", "b", "c", "d"),
        edges=(("a", "b"), ("a", "c"), ("a", "d")),
    )
    original = vertex_deck_isomorphism_profile(vertex_deletion_family(source))
    payload = original.model_dump(mode="json")
    path_class = payload["classes"][1]
    relabelled = {
        **path_class,
        "representative": {
            **path_class["representative"],
            "edges": [["v00", "v01"], ["v00", "v02"]],
        },
        "multiplicity": 1,
        "card_indices": [1],
    }
    remaining = {**path_class, "multiplicity": 2, "card_indices": [2, 3]}
    payload["classes"] = [payload["classes"][0], relabelled, remaining]
    payload["class_indices"] = [0, 1, 2, 2]
    payload["vertex_maps"][1] = [0, 1, 2]
    decoded = (
        VertexDeckIsomorphismProfile.model_validate_json(json.dumps(payload))
        if wire
        else VertexDeckIsomorphismProfile.model_validate(payload)
    )
    assert len(decoded.classes) == 3
    # The admitted operation consumes the retained family, not the authored
    # quotient claim. K1,3 has one empty card and three isomorphic P3 cards.
    recomputed = vertex_deck_isomorphism_profile(decoded.family)
    assert recomputed == original
    assert tuple(item.multiplicity for item in recomputed.classes) == (1, 3)
    assert recomputed.class_indices == (0, 1, 1, 1)


@pytest.mark.parametrize("wire", [False, True])
def test_authored_edge_partition_does_not_authenticate_its_quotient(wire: bool) -> None:
    """All three edge cards of K1,3 belong to the same isomorphism class."""
    source = SimpleUndirectedGraph(
        vertices=("a", "b", "c", "d"),
        edges=(("a", "b"), ("a", "c"), ("a", "d")),
    )
    original = edge_deck_isomorphism_profile(edge_deletion_family(source))
    payload = original.model_dump(mode="json")
    path_class = payload["classes"][0]
    relabelled = {
        **path_class,
        "representative": {
            **path_class["representative"],
            "edges": [["v00", "v02"], ["v00", "v03"]],
        },
        "multiplicity": 1,
        "card_indices": [0],
        "deleted_edges": path_class["deleted_edges"][:1],
    }
    remaining = {
        **path_class,
        "multiplicity": 2,
        "card_indices": [1, 2],
        "deleted_edges": path_class["deleted_edges"][1:],
    }
    payload["classes"] = [relabelled, remaining]
    payload["class_indices"] = [0, 1, 1]
    payload["vertex_maps"][0] = [0, 1, 2, 3]
    decoded = (
        EdgeDeckIsomorphismProfile.model_validate_json(json.dumps(payload))
        if wire
        else EdgeDeckIsomorphismProfile.model_validate(payload)
    )
    assert len(decoded.classes) == 2
    recomputed = edge_deck_isomorphism_profile(decoded.family)
    assert recomputed == original
    assert tuple(item.multiplicity for item in recomputed.classes) == (3,)
    assert recomputed.class_indices == (0, 0, 0)
