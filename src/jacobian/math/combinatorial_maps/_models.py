"""Typed wire contracts for combinatorial-map operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.combinatorial_maps.values import (
    FiniteCombinatorialMap,
)


class FacesRequest(StrictModel):
    """Compute the complete face-orbit family of a combinatorial map."""

    map: FiniteCombinatorialMap


class FacesResult(StrictModel):
    """The complete face-orbit family of the supplied map.

    The original input map is carried on the result so the bound model can
    re-run the exact native kernel and verify the face-orbit family.
    """

    map: FiniteCombinatorialMap
    face_walks: tuple[tuple[int, ...], ...]
    face_of_dart: tuple[int, ...]
    successor: tuple[int, ...]

    @model_validator(mode="after")
    def bind_faces(self) -> Self:
        from jacobian.math.combinatorial_maps.operations_module import face_orbits

        walks, face_of_dart, successor, _ = face_orbits(self.map)
        expected_walks = tuple(tuple(walk) for walk in walks)
        if self.face_walks != expected_walks:
            raise ValueError("face_walks must be the exact face-orbit family")
        n = len(self.map.darts)
        if self.face_of_dart != tuple(face_of_dart[d] for d in range(n)):
            raise ValueError("face_of_dart must be the exact per-dart face assignment")
        if self.successor != tuple(successor):
            raise ValueError("successor must be the exact dart-successor permutation")
        return self


class EulerCharacteristicRequest(StrictModel):
    """Compute per-component and total Euler characteristic."""

    map: FiniteCombinatorialMap


class EulerCharacteristicResult(StrictModel):
    """Per-component and total Euler characteristic."""

    per_component: tuple[dict[str, int], ...]
    total: dict[str, int]

    @model_validator(mode="after")
    def bind_euler(self) -> Self:
        required = {"V", "E", "F", "chi"}
        if set(self.total.keys()) != required:
            raise ValueError("total must carry V, E, F, chi")
        for row in self.per_component:
            if set(row.keys()) != required:
                raise ValueError("each component row must carry V, E, F, chi")
        return self


class OrientableGenusRequest(StrictModel):
    """Compute per-component and total orientable genus."""

    map: FiniteCombinatorialMap


class OrientableGenusResult(StrictModel):
    """Per-component and total orientable genus."""

    per_component: tuple[int, ...]
    total: int = Field(ge=0)


class OrientationReverseRequest(StrictModel):
    """Reverse every local cyclic order of a combinatorial map."""

    map: FiniteCombinatorialMap


class OrientationReverseResult(StrictModel):
    """The orientation-reversed map and the induced face bijection.

    The original input map is carried on the result so the bound model can
    re-run the exact native kernel and verify the reversed map and face
    bijection.
    """

    map: FiniteCombinatorialMap
    reversed_map: FiniteCombinatorialMap
    face_bijection: dict[int, int]

    @model_validator(mode="after")
    def bind_orientation_reverse(self) -> Self:
        from jacobian.math.combinatorial_maps.operations_module import (
            orientation_reverse,
        )

        expected_reversed, expected_bijection = orientation_reverse(self.map)
        if self.reversed_map != expected_reversed:
            raise ValueError(
                "reversed_map must be the exact orientation reversal of the input map"
            )
        if self.face_bijection != expected_bijection:
            raise ValueError("face_bijection must be the exact induced face bijection")
        return self


class ConnectedComponentsRequest(StrictModel):
    """Return the component partition of vertices, darts, and faces."""

    map: FiniteCombinatorialMap


class ConnectedComponentsResult(StrictModel):
    """``vertex -> component``, ``dart -> component``, ``face -> component``."""

    vertex_component: tuple[int, ...]
    dart_component: tuple[int, ...]
    face_component: tuple[int, ...]


class DualRequest(StrictModel):
    """Compute the exact embedded dual of a combinatorial map."""

    map: FiniteCombinatorialMap


class DualResult(StrictModel):
    """The dual combinatorial map and the primal-dart -> dual-dart bijection."""

    dual: FiniteCombinatorialMap
    primal_to_dual: dict[int, int]


class VertexFaceIncidenceRequest(StrictModel):
    """Return the exact incidence structure between vertices and faces."""

    map: FiniteCombinatorialMap


class VertexFaceIncidenceResult(StrictModel):
    """Per-(vertex, face) multiplicity and per-vertex face set.

    ``multiplicity`` maps each vertex to its per-face occurrence counts;
    the nested shape keeps the wire representation JSON-safe.
    """

    multiplicity: dict[int, dict[int, int]]
    boolean_incidence: dict[int, tuple[int, ...]]


__all__ = [
    "ConnectedComponentsRequest",
    "ConnectedComponentsResult",
    "DualRequest",
    "DualResult",
    "EulerCharacteristicRequest",
    "EulerCharacteristicResult",
    "FacesRequest",
    "FacesResult",
    "OrientableGenusRequest",
    "OrientableGenusResult",
    "OrientationReverseRequest",
    "OrientationReverseResult",
    "VertexFaceIncidenceRequest",
    "VertexFaceIncidenceResult",
]
