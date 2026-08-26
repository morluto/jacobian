"""Typed wire contracts for combinatorial-map operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.combinatorial_maps.values import (
    FiniteCombinatorialMap,
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"combinatorial_map.{reason}", message)


class FacesRequest(StrictModel):
    """Compute the complete face-orbit family of a combinatorial map."""

    map: FiniteCombinatorialMap


class FacesResult(StrictModel):
    """The complete face-orbit family of the supplied map.

    The retained map binds the carrier and result envelope.  An independently
    supplied face claim is checked by the explicit owner verifier; parsing
    this wire value never re-enters an operation.
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

    @classmethod
    def _from_kernel(
        cls,
        request: FacesRequest,
        *,
        face_walks: tuple[tuple[int, ...], ...],
        face_of_dart: tuple[int, ...],
        successor: tuple[int, ...],
    ) -> Self:
        return cls.model_construct(
            map=request.map,
            face_walks=face_walks,
            face_of_dart=face_of_dart,
            successor=successor,
        )


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
            raise _validation_error("euler_total_keys", "total must carry V, E, F, chi")
        for row in self.per_component:
            if set(row.keys()) != required:
                raise _validation_error(
                    "euler_component_keys", "each component row must carry V, E, F, chi"
                )
        return self

    @classmethod
    def _from_kernel(
        cls, *, per_component: tuple[dict[str, int], ...], total: dict[str, int]
    ) -> Self:
        return cls.model_construct(per_component=per_component, total=total)


class OrientableGenusRequest(StrictModel):
    """Compute per-component and total orientable genus."""

    map: FiniteCombinatorialMap


class OrientableGenusResult(StrictModel):
    """Per-component and total orientable genus."""

    per_component: tuple[int, ...]
    total: int = Field(ge=0)

    @classmethod
    def _from_kernel(cls, *, per_component: tuple[int, ...], total: int) -> Self:
        return cls.model_construct(per_component=per_component, total=total)


class OrientationReverseRequest(StrictModel):
    """Reverse every local cyclic order of a combinatorial map."""

    map: FiniteCombinatorialMap


class OrientationReverseResult(StrictModel):
    """The orientation-reversed map and the induced face bijection.

    The retained map binds the carrier.  The exact reversal and induced face
    bijection are checked only by the explicit owner verifier.
    """

    map: FiniteCombinatorialMap
    reversed_map: FiniteCombinatorialMap
    face_bijection: dict[int, int]

    @classmethod
    def _from_kernel(
        cls,
        request: OrientationReverseRequest,
        *,
        reversed_map: FiniteCombinatorialMap,
        face_bijection: dict[int, int],
    ) -> Self:
        return cls.model_construct(
            map=request.map,
            reversed_map=reversed_map,
            face_bijection=face_bijection,
        )


class ConnectedComponentsRequest(StrictModel):
    """Return the component partition of vertices, darts, and faces."""

    map: FiniteCombinatorialMap


class ConnectedComponentsResult(StrictModel):
    """``vertex -> component``, ``dart -> component``, ``face -> component``."""

    vertex_component: tuple[int, ...]
    dart_component: tuple[int, ...]
    face_component: tuple[int, ...]

    @classmethod
    def _from_kernel(
        cls,
        *,
        vertex_component: tuple[int, ...],
        dart_component: tuple[int, ...],
        face_component: tuple[int, ...],
    ) -> Self:
        return cls.model_construct(
            vertex_component=vertex_component,
            dart_component=dart_component,
            face_component=face_component,
        )


class DualRequest(StrictModel):
    """Compute the exact embedded dual of a combinatorial map."""

    map: FiniteCombinatorialMap


class DualResult(StrictModel):
    """The dual combinatorial map and the primal-dart -> dual-dart bijection."""

    dual: FiniteCombinatorialMap
    primal_to_dual: dict[int, int]

    @classmethod
    def _from_kernel(
        cls, *, dual: FiniteCombinatorialMap, primal_to_dual: dict[int, int]
    ) -> Self:
        return cls.model_construct(dual=dual, primal_to_dual=primal_to_dual)


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

    @classmethod
    def _from_kernel(
        cls,
        *,
        multiplicity: dict[int, dict[int, int]],
        boolean_incidence: dict[int, tuple[int, ...]],
    ) -> Self:
        return cls.model_construct(
            multiplicity=multiplicity,
            boolean_incidence=boolean_incidence,
        )


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
