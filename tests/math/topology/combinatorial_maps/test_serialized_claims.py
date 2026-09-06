"""Map admission and source-target relations survive serialization."""

import json
from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.topology.combinatorial_maps import (
    FiniteCombinatorialMap,
    connected_components,
    connected_components_vertices,
    dual_map,
    euler_characteristic,
    face_orbits,
    orientable_genus,
    orientation_reverse,
    verify_dual,
    verify_orientation_reverse,
    verify_vertex_face_incidence,
    vertex_face_incidence,
)


def _edge() -> FiniteCombinatorialMap:
    return FiniteCombinatorialMap(
        vertex_count=2, darts=((0, 1, 1), (1, 0, 0)), rotations=((0,), (1,))
    )


@pytest.mark.parametrize(
    "consumer",
    [
        connected_components,
        connected_components_vertices,
        dual_map,
        euler_characteristic,
        face_orbits,
        orientable_genus,
        orientation_reverse,
        vertex_face_incidence,
    ],
)
def test_map_laws_are_consumer_admission(
    consumer: Callable[[FiniteCombinatorialMap], object],
) -> None:
    payload = _edge().model_dump()
    payload["rotations"] = [[1], [0]]
    claim = FiniteCombinatorialMap.model_validate(payload)
    decoded = FiniteCombinatorialMap.model_validate_json(claim.model_dump_json())
    with pytest.raises(OperationDomainValidationError, match="outgoing darts"):
        consumer(decoded)


def test_map_relation_round_trips_and_forgeries() -> None:
    result: Any
    verifier: Callable[[Any], bool]
    for result, verifier in (
        (dual_map(_edge()), verify_dual),
        (orientation_reverse(_edge()), verify_orientation_reverse),
    ):
        assert verifier(type(result).model_validate_json(result.model_dump_json()))
        payload = result.model_dump(mode="json")
        payload["bijection"]["images"][0] = 1024
        with pytest.raises(ValidationError):
            type(result).model_validate(payload)
    result = dual_map(_edge())
    payload = result.model_dump()
    payload["bijection"]["images"] = [1, 0]
    assert not verify_dual(type(result).model_validate(payload))
    incidence = vertex_face_incidence(_edge())
    assert verify_vertex_face_incidence(
        type(incidence).model_validate_json(incidence.model_dump_json())
    )
    payload = incidence.model_dump(mode="json")
    payload["multiplicity"]["entries"][0]["value"] = {"num": "2", "den": "1"}
    assert not verify_vertex_face_incidence(
        type(incidence).model_validate_json(json.dumps(payload))
    )


def test_incidence_preserves_full_map_axes() -> None:
    # 65 disjoint edges have 130 vertices; their dual has 130 faces.
    source = FiniteCombinatorialMap(
        vertex_count=130,
        darts=tuple((i, i ^ 1, i ^ 1) for i in range(130)),
        rotations=tuple((i,) for i in range(130)),
    )
    for map_ in (source, dual_map(source).dual):
        result = vertex_face_incidence(map_)
        assert result.multiplicity.row_count == map_.vertex_count
        assert verify_vertex_face_incidence(
            type(result).model_validate_json(result.model_dump_json())
        )
