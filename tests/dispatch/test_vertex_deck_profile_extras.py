"""Unknown profile fields must be refused without traversing their values."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from pydantic import ValidationError

from jacobian.math.graphs.decks import _models as models
from jacobian.math.graphs.decks import operations
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _profile() -> models.VertexDeckIsomorphismProfile:
    source = SimpleUndirectedGraph(
        vertices=("a", "b", "c", "d"),
        edges=(("a", "b"), ("a", "c"), ("a", "d")),
    )
    return operations.vertex_deck_isomorphism_profile(
        operations.vertex_deletion_family(source)
    )


_EXTRA_PATHS = [
    ("deck",),
    ("deck", "source"),
    ("deck", "cards", 0),
    ("deck", "cards", 0, "card"),
    ("family",),
    ("family", "source"),
    ("family", "cards", 0),
    ("family", "cards", 0, "card"),
    ("classes", 0),
    ("classes", 0, "representative"),
]


@pytest.mark.parametrize("path", _EXTRA_PATHS)
@pytest.mark.parametrize("json_arrays", [False, True])
def test_nested_extras_are_rejected_without_materialization(
    path: tuple[str | int, ...], json_arrays: bool
) -> None:
    class Untraversable(list[object]):
        def __iter__(self) -> Iterator[object]:
            raise AssertionError("forbidden value was traversed")

    profile = _profile()
    payload = profile.model_dump(mode="json" if json_arrays else "python")
    owner: Any = models.VertexDeckIsomorphismProfile
    if path[0] == "deck":
        payload = {"deck": payload["family"]}
        owner = models.VertexDeckIsomorphismProfileRequest
    nested: Any = payload
    for key in path:
        nested = nested[key]
    nested["unknown"] = Untraversable()
    with pytest.raises(ValidationError) as exc_info:
        owner.model_validate(payload)
    assert exc_info.value.errors()[0]["type"] == "extra_forbidden"
    assert exc_info.value.errors()[0]["loc"] == (*path, "unknown")


@pytest.mark.parametrize("path", _EXTRA_PATHS)
def test_nested_json_extras_remain_forbidden(path: tuple[str | int, ...]) -> None:
    payload = _profile().model_dump(mode="json")
    owner: Any = models.VertexDeckIsomorphismProfile
    if path[0] == "deck":
        payload = {"deck": payload["family"]}
        owner = models.VertexDeckIsomorphismProfileRequest
    nested: Any = payload
    for key in path:
        nested = nested[key]
    nested["unknown"] = [[0]]
    with pytest.raises(ValidationError) as exc_info:
        owner.model_validate_json(json.dumps(payload), strict=True)
    assert exc_info.value.errors()[0]["type"] == "extra_forbidden"
    assert exc_info.value.errors()[0]["loc"] == (*path, "unknown")


def test_valid_profile_strict_json_round_trip() -> None:
    profile = _profile()
    assert (
        models.VertexDeckIsomorphismProfile.model_validate_json(
            profile.model_dump_json(), strict=True
        )
        == profile
    )
    assert (
        models.VertexDeckIsomorphismProfileRequest.model_validate_json(
            json.dumps({"deck": profile.family.model_dump(mode="json")}), strict=True
        ).deck
        == profile.family
    )
