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
    EdgeDeckIsomorphismProfile,
    EdgeDeckIsomorphismProfileRequest,
    edge_deck_isomorphism_profile,
    edge_deletion_family,
)
from jacobian.math.graphs.decks import _models as deck_models
from jacobian.math.graphs.decks import operations as deck_operations
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _path4() -> SimpleUndirectedGraph:
    return SimpleUndirectedGraph(
        vertices=("a", "b", "c", "d"),
        edges=(("a", "b"), ("b", "c"), ("c", "d")),
    )


def _assert_maps_are_isomorphisms(profile: EdgeDeckIsomorphismProfile) -> None:
    for card_index, mapping in enumerate(profile.vertex_maps):
        card = profile.family.cards[card_index].card
        representative = profile.classes[
            profile.class_indices[card_index]
        ].representative
        positions = {vertex: index for index, vertex in enumerate(card.vertices)}
        mapped = tuple(
            sorted(
                (
                    f"v{min(mapping[positions[left]], mapping[positions[right]]):02d}",
                    f"v{max(mapping[positions[left]], mapping[positions[right]]):02d}",
                )
                for left, right in card.edges
            )
        )
        assert mapped == representative.edges


def test_path4_edge_cards_keep_multiplicity_and_exact_maps() -> None:
    family = edge_deletion_family(_path4())
    profile = edge_deck_isomorphism_profile(
        EdgeDeckIsomorphismProfileRequest(deck=family)
    )

    assert tuple(sorted(item.multiplicity for item in profile.classes)) == (1, 2)
    assert profile.class_indices[0] == profile.class_indices[2]
    assert profile.class_indices[1] != profile.class_indices[0]
    assert tuple(item.deleted_edges for item in profile.classes) == tuple(
        tuple(family.cards[index].deleted_edge for index in item.card_indices)
        for item in profile.classes
    )
    _assert_maps_are_isomorphisms(profile)


def test_edge_profile_round_trip_checks_map_relations() -> None:
    profile = edge_deck_isomorphism_profile(
        EdgeDeckIsomorphismProfileRequest(deck=edge_deletion_family(_path4()))
    )
    assert (
        EdgeDeckIsomorphismProfile.model_validate_json(profile.model_dump_json())
        == profile
    )
    forged = profile.model_dump(mode="python")
    forged["vertex_maps"] = ((0, 0, 1, 2), *profile.vertex_maps[1:])
    with pytest.raises(ValidationError, match="vertex permutation"):
        EdgeDeckIsomorphismProfile.model_validate(forged)


@pytest.mark.parametrize(
    ("field", "mutate"),
    [
        ("class_indices", lambda value: (False, *value[1:])),
        ("vertex_maps", lambda value: ((False, *value[0][1:]), *value[1:])),
    ],
)
def test_wire_profile_rejects_boolean_index_and_map_values(field, mutate) -> None:
    profile = edge_deck_isomorphism_profile(
        EdgeDeckIsomorphismProfileRequest(deck=edge_deletion_family(_path4()))
    )
    forged = profile.model_dump(mode="python")
    forged[field] = mutate(forged[field])
    with pytest.raises(ValidationError, match="exact integers"):
        EdgeDeckIsomorphismProfile.model_validate(forged)


def test_wire_profile_rejects_boolean_class_multiplicity() -> None:
    profile = edge_deck_isomorphism_profile(
        EdgeDeckIsomorphismProfileRequest(deck=edge_deletion_family(_path4()))
    )
    forged = profile.model_dump(mode="python")
    forged["classes"] = (
        {**forged["classes"][0], "multiplicity": True},
        *forged["classes"][1:],
    )
    with pytest.raises(ValidationError, match="exact integer"):
        EdgeDeckIsomorphismProfile.model_validate(forged)


def test_wire_request_rejects_boolean_retained_edge_count() -> None:
    valid_family = edge_deletion_family(
        SimpleUndirectedGraph(vertices=("a", "b", "c"), edges=(("a", "b"), ("b", "c")))
    )
    cards = (
        valid_family.cards[0].model_copy(update={"retained_edge_count": True}),
        *valid_family.cards[1:],
    )
    family = valid_family.model_copy(update={"cards": cards}).model_dump(mode="python")
    with pytest.raises(ValidationError, match="exact integer"):
        EdgeDeckIsomorphismProfileRequest.model_validate({"deck": family})


def test_edgeless_graph_retains_empty_edge_profile() -> None:
    family = edge_deletion_family(
        SimpleUndirectedGraph(vertices=("a", "b", "c"), edges=())
    )
    profile = edge_deck_isomorphism_profile(
        EdgeDeckIsomorphismProfileRequest(deck=family)
    )
    assert profile.family == family
    assert profile.classes == profile.class_indices == profile.vertex_maps == ()


def test_native_operation_admits_work_before_card_canonicalization(monkeypatch) -> None:
    source = SimpleUndirectedGraph(
        vertices=tuple(f"v{index}" for index in range(10)),
        edges=(("v0", "v1"),),
    )
    request = EdgeDeckIsomorphismProfileRequest.model_construct(
        deck=edge_deletion_family(source)
    )
    monkeypatch.setattr(
        deck_operations,
        "_canonical_card_form",
        lambda *_: pytest.fail("permutation work must be admitted first"),
    )
    with pytest.raises(OperationResourceAdmissionError, match="shared work bound"):
        edge_deck_isomorphism_profile(request)


def test_work_and_output_bounds_have_exact_boundaries(monkeypatch) -> None:
    family = edge_deletion_family(_path4())
    request = EdgeDeckIsomorphismProfileRequest.model_construct(deck=family)
    _, exact_work, exact_bytes = deck_models._edge_iso_profile_resource_estimates(
        4, 3, 3
    )
    monkeypatch.setattr(
        deck_operations, "MAX_UNLABELLED_DECK_ISOMORPHISM_WORK", exact_work
    )
    monkeypatch.setattr(
        deck_operations,
        "MAX_EDGE_DECK_ISOMORPHISM_PROFILE_RESULT_BYTES",
        exact_bytes,
    )
    assert edge_deck_isomorphism_profile(request).classes

    monkeypatch.setattr(
        deck_operations, "MAX_UNLABELLED_DECK_ISOMORPHISM_WORK", exact_work - 1
    )
    with pytest.raises(OperationResourceAdmissionError, match="shared work bound"):
        edge_deck_isomorphism_profile(request)

    monkeypatch.setattr(
        deck_operations, "MAX_UNLABELLED_DECK_ISOMORPHISM_WORK", exact_work
    )
    monkeypatch.setattr(
        deck_operations,
        "MAX_EDGE_DECK_ISOMORPHISM_PROFILE_RESULT_BYTES",
        exact_bytes - 1,
    )
    with pytest.raises(OperationResourceAdmissionError, match="serialized byte bound"):
        edge_deck_isomorphism_profile(request)


def test_raw_tuple_preflight_rejects_order_before_nested_family_parsing() -> None:
    payload = {
        "deck": {
            "source": {
                "vertices": tuple(f"v{index}" for index in range(11)),
                "edges": (),
            },
            "cards": "malformed but over the admitted order",
        }
    }
    with pytest.raises(ValidationError, match="at most 10 source vertices"):
        EdgeDeckIsomorphismProfileRequest.model_validate(payload)


def test_catalog_admission_rejects_before_nested_family_parsing() -> None:
    payload = {
        "deck": {
            "source": {
                "vertices": [f"v{index}" for index in range(11)],
                "edges": [],
            },
            "cards": "malformed but over the admitted order",
        }
    }
    with pytest.raises(OperationRequestValidationError) as error:
        invoke_operation(
            "graph.deck.edge.isomorphism_classes.compute", payload, Catalog.open()
        )
    assert "at most 10 source vertices" in str(error.value.cause)


def test_catalog_producer_does_not_replay_class_canonicalization(monkeypatch) -> None:
    operation = Catalog.open().operation("graph.deck.edge.isomorphism_classes.compute")
    assert operation is not None
    original = deck_models._canonical_card_edges
    calls = 0

    def counted(vertices, edges):
        nonlocal calls
        calls += 1
        return original(vertices, edges)

    with monkeypatch.context() as patcher:
        patcher.setattr(deck_models, "_canonical_card_edges", counted)
        invocation = invoke_operation(
            operation.operation_id, operation.examples[0].input, Catalog.open()
        )
        assert calls == 0
        decoded = EdgeDeckIsomorphismProfile.model_validate_json(
            json.dumps(invocation.output)
        )
        assert calls == 0
    assert sum(item.multiplicity for item in decoded.classes) == 3
    _assert_maps_are_isomorphisms(decoded)


def test_wire_output_shape_bound_precedes_nested_representative_canonicalization(
    monkeypatch,
) -> None:
    profile = edge_deck_isomorphism_profile(
        EdgeDeckIsomorphismProfileRequest(deck=edge_deletion_family(_path4()))
    )
    payload = profile.model_dump(mode="json")
    payload["classes"][0]["representative"]["edges"] *= 100
    monkeypatch.setattr(
        deck_models,
        "_canonical_card_edges",
        lambda *_: pytest.fail(
            "shape admission must precede representative validation"
        ),
    )
    with pytest.raises(ValidationError, match="simple-graph edge bound"):
        EdgeDeckIsomorphismProfile.model_validate(payload)


def test_wire_output_class_count_is_capped_before_row_preflight(monkeypatch) -> None:
    profile = edge_deck_isomorphism_profile(
        EdgeDeckIsomorphismProfileRequest(deck=edge_deletion_family(_path4()))
    )
    payload = profile.model_dump(mode="python")
    payload["classes"] = (*payload["classes"],) * 100
    monkeypatch.setattr(
        deck_models,
        "_preflight_edge_profile_result_rows",
        lambda *_: pytest.fail("class count must be capped before row traversal"),
    )
    with pytest.raises(
        ValidationError, match="more isomorphism classes than edge cards"
    ):
        EdgeDeckIsomorphismProfile.model_validate(payload)


def test_native_operation_rejects_boolean_retained_edge_count() -> None:
    family = edge_deletion_family(_path4())
    forged_card = family.cards[0].model_copy(update={"retained_edge_count": True})
    forged = family.model_copy(update={"cards": (forged_card, *family.cards[1:])})
    request = EdgeDeckIsomorphismProfileRequest.model_construct(deck=forged)
    with pytest.raises(OperationDomainValidationError, match="canonical graph axes"):
        edge_deck_isomorphism_profile(request)


def test_wire_profile_rejects_scalar_deleted_edge_entry() -> None:
    source = SimpleUndirectedGraph(vertices=("a", "b", "c"), edges=(("a", "b"),))
    profile = edge_deck_isomorphism_profile(
        EdgeDeckIsomorphismProfileRequest(deck=edge_deletion_family(source))
    )
    forged = profile.model_dump(mode="python")
    first = forged["classes"][0]
    forged["classes"] = ({**first, "deleted_edges": ["ab"]}, *forged["classes"][1:])
    with pytest.raises(ValidationError):
        EdgeDeckIsomorphismProfile.model_validate(forged)
