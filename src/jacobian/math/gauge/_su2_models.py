"""Typed exact SU(2) edge fields over rational unit quaternions."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.gauge._models import (
    MAX_GAUGE_EDGES,
    MAX_GAUGE_VERTICES,
    GaugeLabel,
    GaugeLattice,
    OrientedGaugePath,
)
from jacobian.math.quaternions._models import RationalUnitQuaternion


def _error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"lattice_gauge.su2.{reason}", message)


class SU2GaugeEdgeValue(StrictModel):
    edge_id: GaugeLabel
    value: RationalUnitQuaternion


class SU2GaugeField(StrictModel):
    """One rational SU(2) link variable on each edge of a gauge lattice."""

    lattice: GaugeLattice
    edge_values: tuple[SU2GaugeEdgeValue, ...] = Field(
        min_length=1, max_length=MAX_GAUGE_EDGES
    )

    @model_validator(mode="after")
    def require_complete_field(self) -> Self:
        expected = tuple(edge.edge_id for edge in self.lattice.edges)
        actual = tuple(value.edge_id for value in self.edge_values)
        if actual != expected:
            raise _error("field_coverage", "SU(2) values must match lattice edge order")
        return self


class SU2GaugeVertexValue(StrictModel):
    vertex: GaugeLabel
    value: RationalUnitQuaternion


class SU2GaugeTransformRequest(StrictModel):
    field: SU2GaugeField
    vertex_values: tuple[SU2GaugeVertexValue, ...] = Field(
        min_length=1, max_length=MAX_GAUGE_VERTICES
    )

    @model_validator(mode="after")
    def require_complete_frames(self) -> Self:
        if (
            tuple(value.vertex for value in self.vertex_values)
            != self.field.lattice.vertices
        ):
            raise _error("transform_vertices", "frames must match lattice vertex order")
        return self


class SU2GaugeTransformResult(StrictModel):
    source: SU2GaugeField
    transformed: SU2GaugeField
    vertex_values: tuple[SU2GaugeVertexValue, ...]


class SU2HolonomyRequest(StrictModel):
    field: SU2GaugeField
    path: OrientedGaugePath


class SU2HolonomyResult(StrictModel):
    field: SU2GaugeField
    path: OrientedGaugePath
    holonomy: RationalUnitQuaternion
    start: str
    end: str


class SU2WilsonTraceResult(StrictModel):
    """The ordinary two-dimensional trace, 2 Re(holonomy), with its parent."""

    holonomy: SU2HolonomyResult
    trace: CanonicalRational


__all__ = [
    "SU2GaugeEdgeValue",
    "SU2GaugeField",
    "SU2GaugeTransformRequest",
    "SU2GaugeTransformResult",
    "SU2GaugeVertexValue",
    "SU2HolonomyRequest",
    "SU2HolonomyResult",
    "SU2WilsonTraceResult",
]
