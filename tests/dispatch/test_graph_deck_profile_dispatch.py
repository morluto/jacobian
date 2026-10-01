"""Strict-JSON dispatch for the vertex-deck isomorphism profile.

A remote caller enters through ``parse_operation_input``, which validates
strictly. These models must stay reachable that way with real producer data,
and no mutable JSON array may survive inside the frozen decoded value.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from jacobian.dispatch import parse_operation_input
from jacobian.math.graphs.decks._models import (
    VertexDeckIsomorphismProfile,
    VertexDeckIsomorphismProfileRequest,
)
from jacobian.math.graphs.decks.operations import (
    vertex_deck_isomorphism_profile,
    vertex_deletion_family,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _source() -> SimpleUndirectedGraph:
    vertices = ("v00", "v01", "v02", "v03")
    return SimpleUndirectedGraph(
        vertices=vertices,
        edges=tuple(
            (vertices[left], vertices[right])
            for left in range(len(vertices))
            for right in range(left + 1, len(vertices))
        ),
    )


def _raw_lists(value: Any, path: str = "root") -> list[str]:
    """Report every JSON array that survived as a mutable ``list``."""

    found: list[str] = []
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(_raw_lists(item, f"{path}[{index}]"))
        if type(value) is list:
            found.append(path)
    elif isinstance(value, dict):
        for key, item in value.items():
            found.extend(_raw_lists(item, f"{path}.{key}"))
    return found


@pytest.mark.parametrize("wire", [False, True])
def test_strict_dispatch_decodes_the_vertex_iso_profile(wire: bool) -> None:
    profile = vertex_deck_isomorphism_profile(vertex_deletion_family(_source()))
    encoded = profile.model_dump_json()
    decoded = parse_operation_input(
        VertexDeckIsomorphismProfile,
        json.loads(encoded) if wire else json.loads(json.dumps(profile.model_dump())),
    )
    assert _raw_lists(decoded) == []
    assert decoded == profile


@pytest.mark.parametrize("wire", [False, True])
def test_strict_dispatch_decodes_the_vertex_iso_profile_request(wire: bool) -> None:
    family = vertex_deletion_family(_source())
    payload = family.model_dump_json() if wire else json.dumps(family.model_dump())
    decoded = parse_operation_input(
        VertexDeckIsomorphismProfileRequest,
        {"deck": json.loads(payload)},
    )
    assert _raw_lists(decoded) == []
    assert decoded.deck == family
