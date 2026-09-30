"""Regressions for the deck profile decode boundary and family carrier.

Result decoding must not replay the producer's canonicalization search, and the
normalizer must bound attacker-controlled rows before copying them.
"""

from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.decks import _models as models_module
from jacobian.math.graphs.decks._models import (
    VertexDeckIsomorphismProfile,
    VertexDeletionFamily,
)
from jacobian.math.graphs.decks.operations import (
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


def test_result_decode_does_not_recanonicalize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Decoding an admitted result must not enumerate vertex permutations."""
    original = _profile()
    calls = 0
    real = models_module._canonical_card_edges

    def counting(vertices, edges):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return real(vertices, edges)

    monkeypatch.setattr(models_module, "_canonical_card_edges", counting)
    VertexDeckIsomorphismProfile.model_validate(original.model_dump())
    assert calls == 0


def test_canonicalization_is_still_used_by_the_producer() -> None:
    """The producer keeps canonicalizing; only result decoding stopped."""
    import inspect

    import jacobian.math.graphs.decks.operations as deck_operations

    source = inspect.getsource(deck_operations)
    assert "_canonical_card_edges(" in source
    assert models_module._canonical_card_edges is not None


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

    def spy(value):  # type: ignore[no-untyped-def]
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

    def spy(value):  # type: ignore[no-untyped-def]
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


def test_profile_round_trips_unchanged() -> None:
    original = _profile()
    again = VertexDeckIsomorphismProfile.model_validate(original.model_dump())
    assert again == original


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
