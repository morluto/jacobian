"""Typed wire contracts for combinatorial-map operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.topology.combinatorial_maps.values import (
    FiniteCombinatorialMap,
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"combinatorial_map.{reason}", message)


class FacesRequest(StrictModel):
    """Compute the complete face-orbit family of a combinatorial map."""

    map: FiniteCombinatorialMap


class FacesResult(StrictModel):
    """The complete face-orbit family of the supplied map.

    The retained map binds the carrier and result envelope. Parsing this wire
    value never re-enters an operation.
    """

    map: FiniteCombinatorialMap
    face_walks: tuple[tuple[int, ...], ...]
    face_of_dart: tuple[int, ...]
    successor: tuple[int, ...]

    @model_validator(mode="after")
    def require_face_result_shape(self) -> Self:
        dart_count = len(self.map.darts)
        if len(self.face_of_dart) != dart_count or len(self.successor) != dart_count:
            raise _validation_error(
                "face_result_dart_count",
                "face assignment and successor must cover the map's darts",
            )
        if (
            any(not walk for walk in self.face_walks)
            or sum(len(walk) for walk in self.face_walks) != dart_count
        ):
            raise _validation_error(
                "face_walk_shape",
                "face walks must be nonempty and have one entry per dart in total",
            )
        if any(
            dart < 0 or dart >= dart_count for walk in self.face_walks for dart in walk
        ) or any(
            dart < 0 or dart >= len(self.face_walks) for dart in self.face_of_dart
        ):
            raise _validation_error(
                "face_result_index_out_of_range",
                "face-result indices must lie in their declared carriers",
            )
        if any(dart < 0 or dart >= dart_count for dart in self.successor):
            raise _validation_error(
                "successor_out_of_range",
                "successor entries must be dart indices",
            )
        return self


class EulerCharacteristicRequest(StrictModel):
    """Compute per-component and total Euler characteristic."""

    map: FiniteCombinatorialMap


class EulerCharacteristicCounts(StrictModel):
    """Cell counts and their Euler characteristic for one component family."""

    vertices: int = Field(ge=0)
    edges: int = Field(ge=0)
    faces: int = Field(ge=0)
    characteristic: int

    @model_validator(mode="after")
    def bind_characteristic(self) -> Self:
        if self.characteristic != self.vertices - self.edges + self.faces:
            raise _validation_error(
                "euler_characteristic_mismatch",
                "characteristic must equal vertices - edges + faces",
            )
        return self


class EulerCharacteristicResult(StrictModel):
    """Per-component and total Euler characteristic."""

    per_component: tuple[EulerCharacteristicCounts, ...]
    total: EulerCharacteristicCounts


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

    The retained map binds the carrier. The exact reversal and induced face
    bijection are established by the defining operation.
    """

    map: FiniteCombinatorialMap
    reversed_map: FiniteCombinatorialMap
    face_bijection: dict[int, int]


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
    "EulerCharacteristicCounts",
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
