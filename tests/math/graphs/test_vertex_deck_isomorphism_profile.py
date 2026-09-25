from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.dispatch import OperationRequestValidationError, invoke_operation
from jacobian.math.graphs.decks import (
    VertexDeckIsomorphismProfile,
    VertexDeckIsomorphismProfileRequest,
    vertex_deck_isomorphism_profile,
    vertex_deletion_family,
)
from jacobian.math.graphs.decks import _models as deck_models
from jacobian.math.graphs.decks import operations as deck_operations
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _path3(prefix: str = "") -> SimpleUndirectedGraph:
    vertices = tuple(f"{prefix}{label}" for label in ("a", "b", "c"))
    return SimpleUndirectedGraph(
        vertices=vertices,
        edges=((vertices[0], vertices[1]), (vertices[1], vertices[2])),
    )


def _assert_maps_are_isomorphisms(profile: VertexDeckIsomorphismProfile) -> None:
    for card_index, card_map in enumerate(profile.vertex_maps):
        card = profile.family.cards[card_index].card
        representative = profile.classes[
            profile.class_indices[card_index]
        ].representative
        mapped = tuple(
            sorted(
                (
                    f"v{min(card_map[card.vertices.index(left)], card_map[card.vertices.index(right)]):02d}",
                    f"v{max(card_map[card.vertices.index(left)], card_map[card.vertices.index(right)]):02d}",
                )
                for left, right in card.edges
            )
        )
        assert mapped == representative.edges


def test_path3_profile_groups_cards_and_returns_exact_vertex_maps() -> None:
    family = vertex_deletion_family(_path3())
    profile = vertex_deck_isomorphism_profile(
        VertexDeckIsomorphismProfileRequest(deck=family)
    )
    assert profile.class_indices == (1, 0, 1)
    assert tuple(item.multiplicity for item in profile.classes) == (1, 2)
    assert tuple(item.card_indices for item in profile.classes) == ((1,), (0, 2))
    assert tuple(item.representative.edges for item in profile.classes) == (
        (),
        (("v00", "v01"),),
    )
    assert profile.vertex_maps == ((0, 1), (0, 1), (0, 1))
    _assert_maps_are_isomorphisms(profile)


def test_card_relabelling_preserves_classes_and_map_relations() -> None:
    first = vertex_deck_isomorphism_profile(
        VertexDeckIsomorphismProfileRequest(deck=vertex_deletion_family(_path3("x")))
    )
    second = vertex_deck_isomorphism_profile(
        VertexDeckIsomorphismProfileRequest(deck=vertex_deletion_family(_path3("z")))
    )
    assert first.class_indices == second.class_indices
    assert tuple(row.representative for row in first.classes) == tuple(
        row.representative for row in second.classes
    )
    assert tuple(row.multiplicity for row in first.classes) == tuple(
        row.multiplicity for row in second.classes
    )
    _assert_maps_are_isomorphisms(first)
    _assert_maps_are_isomorphisms(second)


@pytest.mark.parametrize("vertices", [(), ("v",)])
def test_zero_order_cards_retain_empty_maps(vertices: tuple[str, ...]) -> None:
    family = vertex_deletion_family(SimpleUndirectedGraph(vertices=vertices, edges=()))
    profile = vertex_deck_isomorphism_profile(
        VertexDeckIsomorphismProfileRequest(deck=family)
    )
    assert len(profile.family.cards) == len(vertices)
    if not vertices:
        assert profile.classes == profile.vertex_maps == profile.class_indices == ()
    else:
        assert len(profile.classes) == 1
        assert profile.classes[0].multiplicity == 1
        assert profile.classes[0].representative.vertices == ()
        assert profile.vertex_maps == ((),)
        assert profile.class_indices == (0,)


def test_profile_round_trip_validates_maps_and_rejects_forged_bijection() -> None:
    profile = vertex_deck_isomorphism_profile(
        VertexDeckIsomorphismProfileRequest(deck=vertex_deletion_family(_path3()))
    )
    assert (
        VertexDeckIsomorphismProfile.model_validate_json(profile.model_dump_json())
        == profile
    )
    forged = profile.model_dump(mode="python")
    forged["vertex_maps"] = ((0, 0), (0, 1), (0, 1))
    with pytest.raises(ValidationError, match="vertex permutation"):
        VertexDeckIsomorphismProfile.model_validate(forged)


def test_native_operation_admits_and_checks_family_before_canonicalization(
    monkeypatch,
) -> None:
    family = vertex_deletion_family(_path3())
    forged = family.model_copy(update={"cards": family.cards[:-1]})
    request = VertexDeckIsomorphismProfileRequest.model_construct(deck=forged)
    monkeypatch.setattr(
        deck_operations,
        "_canonical_card_form",
        lambda *_: pytest.fail("canonicalization must follow family validation"),
    )
    with pytest.raises(
        OperationDomainValidationError, match="one card per source vertex"
    ):
        vertex_deck_isomorphism_profile(request)


@pytest.mark.parametrize("field", ["edge_appearances", "vertex_appearances"])
def test_native_operation_rejects_boolean_appearance_counts(field: str) -> None:
    family = vertex_deletion_family(_path3())
    counts = list(getattr(family, field))
    counts[0] = True
    forged = family.model_copy(update={field: tuple(counts)})
    request = VertexDeckIsomorphismProfileRequest.model_construct(deck=forged)
    with pytest.raises(OperationDomainValidationError, match="appearance ledgers"):
        vertex_deck_isomorphism_profile(request)


@pytest.mark.parametrize("field", ["retained_edge_count", "deleted_edge_count"])
def test_native_operation_rejects_boolean_card_edge_counts(field: str) -> None:
    family = vertex_deletion_family(_path3())
    forged_card = family.cards[0].model_copy(update={field: True})
    forged = family.model_copy(update={"cards": (forged_card, *family.cards[1:])})
    request = VertexDeckIsomorphismProfileRequest.model_construct(deck=forged)
    with pytest.raises(
        OperationDomainValidationError, match="bound source vertex deletion"
    ):
        vertex_deck_isomorphism_profile(request)


def test_work_and_output_admission_have_exact_boundaries(monkeypatch) -> None:
    family = vertex_deletion_family(_path3())
    request = VertexDeckIsomorphismProfileRequest.model_construct(deck=family)
    _, exact_work, exact_output = deck_models._vertex_iso_profile_resource_estimates(
        3, 2, 3
    )
    monkeypatch.setattr(
        deck_operations, "MAX_UNLABELLED_DECK_ISOMORPHISM_WORK", exact_work
    )
    monkeypatch.setattr(
        deck_operations,
        "MAX_VERTEX_DECK_ISOMORPHISM_PROFILE_RESULT_BYTES",
        exact_output,
    )
    assert vertex_deck_isomorphism_profile(request).classes
    monkeypatch.setattr(
        deck_operations, "MAX_UNLABELLED_DECK_ISOMORPHISM_WORK", exact_work - 1
    )
    with pytest.raises(OperationResourceAdmissionError, match="shared work bound"):
        vertex_deck_isomorphism_profile(request)
    monkeypatch.setattr(
        deck_operations, "MAX_UNLABELLED_DECK_ISOMORPHISM_WORK", exact_work
    )
    monkeypatch.setattr(
        deck_operations,
        "MAX_VERTEX_DECK_ISOMORPHISM_PROFILE_RESULT_BYTES",
        exact_output - 1,
    )
    with pytest.raises(OperationResourceAdmissionError, match="serialized byte bound"):
        vertex_deck_isomorphism_profile(request)


def test_serialized_profile_admits_representative_validation_before_canonicalizing(
    monkeypatch,
) -> None:
    profile = vertex_deck_isomorphism_profile(
        VertexDeckIsomorphismProfileRequest(deck=vertex_deletion_family(_path3()))
    )
    data = json.dumps(profile.model_dump(mode="json"))
    _, exact_work, _ = deck_models._vertex_iso_profile_value_resource_estimates(
        3, 2, len(profile.classes)
    )
    monkeypatch.setattr(deck_models, "MAX_UNLABELLED_DECK_ISOMORPHISM_WORK", exact_work)
    assert VertexDeckIsomorphismProfile.model_validate_json(data) == profile

    original = deck_models._canonical_card_edges
    calls = 0

    def counted(vertices, edges):
        nonlocal calls
        calls += 1
        return original(vertices, edges)

    monkeypatch.setattr(deck_models, "_canonical_card_edges", counted)
    monkeypatch.setattr(
        deck_models, "MAX_UNLABELLED_DECK_ISOMORPHISM_WORK", exact_work - 1
    )
    with pytest.raises(ValidationError, match="shared validation work bound"):
        VertexDeckIsomorphismProfile.model_validate_json(data)
    assert calls == 0


def test_catalog_admission_occurs_before_nested_card_parsing() -> None:
    tool_id = "graph.deck.isomorphism_classes.compute"
    source_vertices = [f"v{index}" for index in range(11)]
    payload = {
        "deck": {
            "source": {"vertices": source_vertices, "edges": []},
            "cards": "malformed but over the admitted order",
        }
    }
    operation = Catalog.open().operation(tool_id)
    assert operation is not None
    with pytest.raises(OperationRequestValidationError) as error:
        invoke_operation(tool_id, payload, Catalog.open())
    assert "supports at most 10 source vertices" in str(error.value.cause)


def test_native_request_tuple_preflight_occurs_before_nested_parsing() -> None:
    payload = {
        "deck": {
            "source": {
                "vertices": tuple(f"v{index}" for index in range(11)),
                "edges": (),
            },
            "cards": "malformed but over the admitted order",
        }
    }
    with pytest.raises(ValidationError, match="supports at most 10 source vertices"):
        VertexDeckIsomorphismProfileRequest.model_validate(payload)


def test_catalog_example_executes_and_returns_exact_maps() -> None:
    operation = Catalog.open().operation("graph.deck.isomorphism_classes.compute")
    assert operation is not None
    original = deck_models._canonical_card_edges
    calls = 0

    def counted(vertices, edges):
        nonlocal calls
        calls += 1
        return original(vertices, edges)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(deck_models, "_canonical_card_edges", counted)
        invocation = invoke_operation(
            operation.operation_id, operation.examples[0].input, Catalog.open()
        )
        assert calls == 0
        result = VertexDeckIsomorphismProfile.model_validate_json(
            json.dumps(invocation.output)
        )
        assert calls == 0
    assert result.class_indices == (1, 0, 1)
    _assert_maps_are_isomorphisms(result)
