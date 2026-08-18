"""Typed wire contracts for exact finite-topology operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.finite_topology.operations import (
    beat_points,
    connected_components,
    continuity,
    is_t0,
    specialization_preorder,
)
from jacobian.math.finite_topology.values import (
    BeatPointWitness,
    FiniteTopology,
    PointMap,
)


class SpecializationPreorderRequest(StrictModel):
    topology: FiniteTopology


class SpecializationPreorderResult(SpecializationPreorderRequest):
    relation: tuple[tuple[bool, ...], ...]
    orientation: Literal["RELATION_X_Y_MEANS_X_IN_CLOSURE_OF_SINGLETON_Y"] = (
        "RELATION_X_Y_MEANS_X_IN_CLOSURE_OF_SINGLETON_Y"
    )
    complete: Literal[True] = True
    method: Literal["EXACT_OPEN_NEIGHBORHOOD_CONTAINMENT"] = (
        "EXACT_OPEN_NEIGHBORHOOD_CONTAINMENT"
    )

    @model_validator(mode="after")
    def bind_preorder(self) -> Self:
        if self.relation != specialization_preorder(self.topology):
            raise ValueError("specialization preorder is not bound to the topology")
        return self


class ConnectedComponentsRequest(StrictModel):
    topology: FiniteTopology


class ConnectedComponentsResult(ConnectedComponentsRequest):
    components: tuple[tuple[int, ...], ...]
    component_count: int = Field(ge=1)
    complete: Literal[True] = True
    method: Literal["UNDIRECTED_SPECIALIZATION_COMPARABILITY"] = (
        "UNDIRECTED_SPECIALIZATION_COMPARABILITY"
    )

    @model_validator(mode="after")
    def bind_components(self) -> Self:
        expected = connected_components(self.topology)
        if self.components != expected or self.component_count != len(expected):
            raise ValueError("connected components are not bound to the topology")
        return self


class ContinuityRequest(StrictModel):
    domain: FiniteTopology
    codomain: FiniteTopology
    point_map: PointMap

    @model_validator(mode="after")
    def bind_map_carriers(self) -> Self:
        if self.point_map.domain_point_count != self.domain.point_count:
            raise ValueError("map domain size must match the domain topology")
        if self.point_map.codomain_point_count != self.codomain.point_count:
            raise ValueError("map codomain size must match the codomain topology")
        return self


class ContinuityResult(ContinuityRequest):
    is_continuous: bool
    violating_open_set: tuple[int, ...] | None
    violating_preimage: tuple[int, ...] | None
    method: Literal["EXACT_OPEN_SET_PREIMAGE_CHECK"] = "EXACT_OPEN_SET_PREIMAGE_CHECK"

    @model_validator(mode="after")
    def bind_continuity_analysis(self) -> Self:
        expected = continuity(self.domain, self.codomain, self.point_map)
        if (
            self.is_continuous != expected.is_continuous
            or self.violating_open_set != expected.violating_open_set
            or self.violating_preimage != expected.violating_preimage
        ):
            raise ValueError("continuity result is not bound to the requested map")
        return self


class BeatPointsRequest(StrictModel):
    topology: FiniteTopology

    @model_validator(mode="after")
    def require_t0_semantics(self) -> Self:
        if not is_t0(self.topology):
            raise ValueError("beat-point computation requires a T0 topology")
        return self


class BeatPointsResult(BeatPointsRequest):
    down_beat_points: tuple[BeatPointWitness, ...]
    up_beat_points: tuple[BeatPointWitness, ...]
    complete: Literal[True] = True
    convention: Literal["STRICT_SPECIALIZATION_ORDER_WITH_EXTREMUM_WITNESS"] = (
        "STRICT_SPECIALIZATION_ORDER_WITH_EXTREMUM_WITNESS"
    )
    method: Literal["EXACT_STRICT_ORDER_EXTREMA"] = "EXACT_STRICT_ORDER_EXTREMA"

    @model_validator(mode="after")
    def bind_beat_points(self) -> Self:
        expected = beat_points(self.topology)
        if (
            self.down_beat_points != expected.down_beat_points
            or self.up_beat_points != expected.up_beat_points
        ):
            raise ValueError("beat-point result is not bound to the topology")
        return self


__all__ = [
    "BeatPointsRequest",
    "BeatPointsResult",
    "ConnectedComponentsRequest",
    "ConnectedComponentsResult",
    "ContinuityRequest",
    "ContinuityResult",
    "SpecializationPreorderRequest",
    "SpecializationPreorderResult",
]
