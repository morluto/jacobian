"""Behavioral controls for the original deck trust-boundary findings."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.decks import _models as models
from jacobian.math.graphs.decks import operations
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _star_profile() -> models.VertexDeckIsomorphismProfile:
    graph = SimpleUndirectedGraph(
        vertices=("a", "b", "c", "d"),
        edges=(("a", "b"), ("a", "c"), ("a", "d")),
    )
    return operations.vertex_deck_isomorphism_profile(
        operations.vertex_deletion_family(graph)
    )


@pytest.mark.parametrize("wire", [False, True])
@pytest.mark.parametrize(
    "field", ["multiplicity", "card_indices", "class_indices", "vertex_maps"]
)
@pytest.mark.parametrize("replacement", ["bool", "float", "string"])
def test_profile_refuses_coercible_scalars(
    wire: bool, field: str, replacement: str
) -> None:
    payload = _star_profile().model_dump(mode="json")
    expected = 1 if field == "multiplicity" else 0
    scalar: Any = {
        "bool": bool(expected),
        "float": float(expected),
        "string": str(expected),
    }[replacement]
    if field == "multiplicity":
        payload["classes"][0][field] = scalar
    elif field == "card_indices":
        payload["classes"][0][field][0] = scalar
    elif field == "class_indices":
        payload[field][0] = scalar
    else:
        payload[field][0][0] = scalar
    with pytest.raises(ValueError):
        if wire:
            models.VertexDeckIsomorphismProfile.model_validate_json(json.dumps(payload))
        else:
            models.VertexDeckIsomorphismProfile.model_validate(payload)


@pytest.mark.parametrize("wire", [False, True])
@pytest.mark.parametrize("request_payload", [False, True])
@pytest.mark.parametrize(
    "field",
    [
        "edge_appearances",
        "vertex_appearances",
        "source_edges",
        "cards",
        "card_vertices",
        "card_edges",
        "retained_vertices",
    ],
)
def test_retained_family_axes_are_bounded_before_normalization(
    monkeypatch: pytest.MonkeyPatch, wire: bool, request_payload: bool, field: str
) -> None:
    profile = _star_profile()
    payload = (
        {"deck": profile.family.model_dump(mode="json")}
        if request_payload
        else profile.model_dump(mode="json")
    )
    family = payload["deck" if request_payload else "family"]
    if field == "source_edges":
        family["source"]["edges"] = [["a", "b"]] * 50_000
    elif field == "cards":
        family["cards"] = [family["cards"][0]] * 50_000
    elif field == "card_vertices":
        family["cards"][0]["card"]["vertices"] = ["b"] * 50_000
    elif field == "card_edges":
        family["cards"][0]["card"]["edges"] = [["b", "c"]] * 50_000
    elif field == "retained_vertices":
        family["cards"][0][field] = ["b"] * 50_000
    else:
        family[field] = [1] * 50_000
    encoded = json.dumps(payload)
    owner = (
        models.VertexDeckIsomorphismProfileRequest
        if request_payload
        else models.VertexDeckIsomorphismProfile
    )
    reached = False
    real = models._normalize_vertex_family_json

    def spy(value: dict[str, Any]) -> dict[str, Any]:
        nonlocal reached
        reached = True
        return real(value)

    monkeypatch.setattr(models, "_normalize_vertex_family_json", spy)
    with pytest.raises(ValueError):
        if wire:
            owner.model_validate_json(encoded)
        else:
            owner.model_validate(payload)
    assert not reached


def test_unknown_payload_does_not_trigger_recursive_materialization() -> None:
    class Untraversable(list[object]):
        def __iter__(self) -> Iterator[object]:
            raise AssertionError("unexpected recursive traversal of a forbidden field")

    payload = _star_profile().model_dump()
    payload["unknown"] = Untraversable()
    with pytest.raises(ValueError, match="Extra inputs"):
        models.VertexDeckIsomorphismProfile.model_validate(payload)


@pytest.mark.parametrize("request_payload", [False, True])
def test_missing_source_axis_is_rejected_before_copying(
    monkeypatch: pytest.MonkeyPatch, request_payload: bool
) -> None:
    payload = _star_profile().model_dump(mode="json")
    family = payload["family"]
    family["source"] = {}
    family["cards"] = [family["cards"][0]] * 50_000
    if request_payload:
        payload = {"deck": family}

    def fail(value: Any) -> Any:
        raise AssertionError("an incomplete source must not reach normalization")

    monkeypatch.setattr(models, "_normalize_vertex_family_json", fail)
    owner = (
        models.VertexDeckIsomorphismProfileRequest
        if request_payload
        else models.VertexDeckIsomorphismProfile
    )
    with pytest.raises(ValueError, match="source vertex axis"):
        owner.model_validate_json(json.dumps(payload))


def test_mixed_native_source_carrier_remains_accepted() -> None:
    profile = _star_profile()
    payload = profile.model_dump()
    payload["family"]["source"] = profile.family.source
    assert models.VertexDeckIsomorphismProfile.model_validate(payload) == profile
    assert (
        models.VertexDeckIsomorphismProfileRequest.model_validate(
            {"deck": payload["family"]}
        ).deck
        == profile.family
    )


@pytest.mark.parametrize("consumer", ["ordinary", "induced", "edges"])
@pytest.mark.parametrize(
    "forgery", ["missing", "wrong", "incomplete", "source", "cards", "foreign"]
)
def test_count_consumers_refuse_incomplete_retained_families(
    consumer: str, forgery: str
) -> None:
    class ForeignFamily:
        @property
        def source(self) -> object:
            raise AssertionError("foreign carrier property must not be evaluated")

    family = _star_profile().family
    if forgery == "missing":
        deck = models.UnlabelledVertexDeck.model_construct()
    else:
        forged: Any = {
            "wrong": None,
            "incomplete": models.VertexDeletionFamily.model_construct(),
            "source": family.model_copy(
                update={"source": SimpleUndirectedGraph.model_construct()}
            ),
            "cards": family.model_copy(update={"cards": None}),
            "foreign": ForeignFamily(),
        }[forgery]
        deck = models.UnlabelledVertexDeck.model_construct(family=forged, classes=())
    pattern = SimpleUndirectedGraph(vertices=(), edges=())
    with pytest.raises(OperationDomainValidationError):
        if consumer == "ordinary":
            operations.vertex_deck_subgraph_count(deck, pattern)
        elif consumer == "induced":
            operations.vertex_deck_induced_subgraph_count(deck, pattern)
        else:
            operations.vertex_deck_edge_count(deck)


def test_decoded_star_deck_feeds_all_count_consumers() -> None:
    original = operations.unlabelled_vertex_deck(_star_profile().family)
    deck = models.UnlabelledVertexDeck.model_validate_json(original.model_dump_json())
    point = SimpleUndirectedGraph(vertices=("p",), edges=())
    assert operations.vertex_deck_subgraph_count(deck, point).occurrence_count == 4
    assert (
        operations.vertex_deck_induced_subgraph_count(deck, point).occurrence_count == 4
    )
    assert operations.vertex_deck_edge_count(deck).source_edge_count == 3


def test_native_family_relation_is_still_checked() -> None:
    original = _star_profile()
    forged = original.family.model_copy(update={"edge_appearances": (99, 99, 99)})
    with pytest.raises(ValueError, match="family_edge_receipt"):
        models.VertexDeckIsomorphismProfile(
            family=forged,
            classes=original.classes,
            class_indices=original.class_indices,
            vertex_maps=original.vertex_maps,
        )


@pytest.mark.parametrize("request_payload", [False, True])
@pytest.mark.parametrize("wire", [False, True])
@pytest.mark.parametrize("missing_graph", [False, True])
def test_retained_axis_is_bounded_even_with_malformed_nested_graph(
    monkeypatch: pytest.MonkeyPatch,
    request_payload: bool,
    wire: bool,
    missing_graph: bool,
) -> None:
    payload = _star_profile().model_dump(mode="json")
    family = payload["family"]
    card = family["cards"][0]
    card["retained_vertices"] = ["b"] * 50_000
    if missing_graph:
        del card["card"]
    else:
        card["card"] = None
    if request_payload:
        payload = {"deck": family}
    owner = (
        models.VertexDeckIsomorphismProfileRequest
        if request_payload
        else models.VertexDeckIsomorphismProfile
    )

    reached = False

    def fail(value: Any) -> Any:
        nonlocal reached
        reached = True
        raise AssertionError("oversized retained axis reached normalization")

    monkeypatch.setattr(models, "_normalize_vertex_family_json", fail)
    with pytest.raises(ValueError):
        if wire:
            owner.model_validate_json(json.dumps(payload))
        else:
            owner.model_validate(payload)
    assert not reached
