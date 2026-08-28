"""Typed wire contracts for exact finite-topology operations."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, WithJsonSchema, model_validator
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.topology.finite.open_sets.values import (
    BeatPointWitness,
    FiniteTopology,
    PointMap,
)

# These bounds are the shared execution envelope of the four public finite
# topology operations below.  They are deliberately not carried by
# ``FiniteTopology`` or ``PointMap``: those canonical values are also useful to
# native consumers whose work is bounded differently.
MAX_TOPOLOGY_OPERATION_POINTS = 32
MAX_TOPOLOGY_OPERATION_OPENS = 1_024


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"finite_topology.{reason}", message)


def _topology_operation_schema() -> JsonSchemaValue:
    """Project the wire-operation envelope onto the shared topology value."""

    schema = FiniteTopology.model_json_schema()
    schema["description"] = (
        "A finite topology accepted by these exact operations: at most "
        f"{MAX_TOPOLOGY_OPERATION_POINTS} points and "
        f"{MAX_TOPOLOGY_OPERATION_OPENS} open sets."
    )
    schema["properties"]["point_count"].update(
        maximum=MAX_TOPOLOGY_OPERATION_POINTS,
    )
    schema["properties"]["open_sets"].update(maxItems=MAX_TOPOLOGY_OPERATION_OPENS)
    return schema


TopologyOperationInput = Annotated[
    FiniteTopology,
    WithJsonSchema(_topology_operation_schema()),
]


def _point_map_operation_schema() -> JsonSchemaValue:
    """Expose the continuity envelope without narrowing ``PointMap`` itself."""

    schema = PointMap.model_json_schema()
    schema["description"] = (
        "A total point map whose domain and codomain are within the "
        f"{MAX_TOPOLOGY_OPERATION_POINTS}-point finite-topology operation envelope."
    )
    for field in ("domain_point_count", "codomain_point_count"):
        schema["properties"][field].update(maximum=MAX_TOPOLOGY_OPERATION_POINTS)
    schema["properties"]["values"].update(maxItems=MAX_TOPOLOGY_OPERATION_POINTS)
    return schema


TopologyOperationPointMap = Annotated[
    PointMap,
    WithJsonSchema(_point_map_operation_schema()),
]


def _require_canonical_subset(
    subset: tuple[int, ...], *, point_count: int, label: str
) -> None:
    if tuple(sorted(set(subset))) != subset:
        raise _validation_error(
            f"{label}_not_canonical",
            f"{label} must be sorted with distinct points",
        )
    if any(not 0 <= point < point_count for point in subset):
        raise _validation_error(
            f"{label}_point_out_of_range",
            f"{label} contains a point outside its carrier",
        )


class SpecializationPreorderRequest(StrictModel):
    topology: TopologyOperationInput


class SpecializationPreorderResult(SpecializationPreorderRequest):
    relation: tuple[tuple[bool, ...], ...]
    orientation: Literal["RELATION_X_Y_MEANS_X_IN_CLOSURE_OF_SINGLETON_Y"] = (
        "RELATION_X_Y_MEANS_X_IN_CLOSURE_OF_SINGLETON_Y"
    )

    @model_validator(mode="after")
    def require_relation_shape(self) -> Self:
        order = self.topology.point_count
        if len(self.relation) != order or any(
            len(row) != order for row in self.relation
        ):
            raise _validation_error(
                "specialization_preorder_shape",
                "specialization preorder must be a square relation on the carrier",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: SpecializationPreorderRequest,
        relation: tuple[tuple[bool, ...], ...],
    ) -> Self:
        return cls.model_construct(
            topology=request.topology,
            relation=relation,
            orientation="RELATION_X_Y_MEANS_X_IN_CLOSURE_OF_SINGLETON_Y",
        )


class ConnectedComponentsRequest(StrictModel):
    topology: TopologyOperationInput


class ConnectedComponentsResult(ConnectedComponentsRequest):
    components: tuple[tuple[int, ...], ...]
    component_count: int = Field(ge=1)

    @model_validator(mode="after")
    def require_component_partition_shape(self) -> Self:
        if self.component_count != len(self.components):
            raise _validation_error(
                "connected_components_count",
                "component count must equal the number of components",
            )
        points: list[int] = []
        for component in self.components:
            _require_canonical_subset(
                component,
                point_count=self.topology.point_count,
                label="connected_component",
            )
            if not component:
                raise _validation_error(
                    "connected_component_empty", "components must be nonempty"
                )
            points.extend(component)
        if tuple(sorted(points)) != tuple(range(self.topology.point_count)):
            raise _validation_error(
                "connected_components_not_partition",
                "components must be a disjoint partition of the topology carrier",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: ConnectedComponentsRequest,
        components: tuple[tuple[int, ...], ...],
    ) -> Self:
        return cls.model_construct(
            topology=request.topology,
            components=components,
            component_count=len(components),
        )


class ContinuityRequest(StrictModel):
    domain: TopologyOperationInput
    codomain: TopologyOperationInput
    point_map: TopologyOperationPointMap

    @model_validator(mode="after")
    def bind_map_carriers(self) -> Self:
        if self.point_map.domain_point_count != self.domain.point_count:
            raise _validation_error(
                "map_domain_size_mismatch",
                "map domain size must match the domain topology",
            )
        if self.point_map.codomain_point_count != self.codomain.point_count:
            raise _validation_error(
                "map_codomain_size_mismatch",
                "map codomain size must match the codomain topology",
            )
        return self


class ContinuityResult(ContinuityRequest):
    is_continuous: bool
    violating_open_set: tuple[int, ...] | None
    violating_preimage: tuple[int, ...] | None

    @model_validator(mode="after")
    def require_witness_shape(self) -> Self:
        if self.is_continuous:
            if (
                self.violating_open_set is not None
                or self.violating_preimage is not None
            ):
                raise _validation_error(
                    "continuous_result_has_witness",
                    "a continuous map cannot carry a violating-open-set witness",
                )
            return self
        if self.violating_open_set is None or self.violating_preimage is None:
            raise _validation_error(
                "discontinuous_result_missing_witness",
                "a discontinuous result requires an open-set and preimage witness",
            )
        _require_canonical_subset(
            self.violating_open_set,
            point_count=self.codomain.point_count,
            label="violating_open_set",
        )
        _require_canonical_subset(
            self.violating_preimage,
            point_count=self.domain.point_count,
            label="violating_preimage",
        )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: ContinuityRequest,
        *,
        is_continuous: bool,
        violating_open_set: tuple[int, ...] | None,
        violating_preimage: tuple[int, ...] | None,
    ) -> Self:
        return cls.model_construct(
            domain=request.domain,
            codomain=request.codomain,
            point_map=request.point_map,
            is_continuous=is_continuous,
            violating_open_set=violating_open_set,
            violating_preimage=violating_preimage,
        )


class BeatPointsRequest(StrictModel):
    topology: TopologyOperationInput


class BeatPointsResult(BeatPointsRequest):
    down_beat_points: tuple[BeatPointWitness, ...]
    up_beat_points: tuple[BeatPointWitness, ...]
    convention: Literal["STRICT_SPECIALIZATION_ORDER_WITH_EXTREMUM_WITNESS"] = (
        "STRICT_SPECIALIZATION_ORDER_WITH_EXTREMUM_WITNESS"
    )

    @model_validator(mode="after")
    def require_witness_shape(self) -> Self:
        for witnesses, label in (
            (self.down_beat_points, "down_beat_points"),
            (self.up_beat_points, "up_beat_points"),
        ):
            points = tuple(witness.point for witness in witnesses)
            if points != tuple(sorted(set(points))):
                raise _validation_error(
                    f"{label}_not_canonical",
                    f"{label} must be sorted with one witness per point",
                )
            for witness in witnesses:
                if (
                    witness.point >= self.topology.point_count
                    or witness.witness >= self.topology.point_count
                    or witness.point == witness.witness
                ):
                    raise _validation_error(
                        f"{label}_invalid_witness",
                        f"{label} witnesses must be distinct topology points",
                    )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: BeatPointsRequest,
        *,
        down_beat_points: tuple[BeatPointWitness, ...],
        up_beat_points: tuple[BeatPointWitness, ...],
    ) -> Self:
        return cls.model_construct(
            topology=request.topology,
            down_beat_points=down_beat_points,
            up_beat_points=up_beat_points,
            convention="STRICT_SPECIALIZATION_ORDER_WITH_EXTREMUM_WITNESS",
        )


__all__ = [
    "MAX_TOPOLOGY_OPERATION_OPENS",
    "MAX_TOPOLOGY_OPERATION_POINTS",
    "BeatPointsRequest",
    "BeatPointsResult",
    "ConnectedComponentsRequest",
    "ConnectedComponentsResult",
    "ContinuityRequest",
    "ContinuityResult",
    "SpecializationPreorderRequest",
    "SpecializationPreorderResult",
]
