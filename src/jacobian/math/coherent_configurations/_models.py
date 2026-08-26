"""Wire contracts for exact coherent-configuration analysis."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.coherent_configurations.values import (
    MAX_COHERENT_CONFIGURATION_POINTS,
    MAX_COHERENT_CONFIGURATION_RELATIONS,
    CoherentConfigurationInput,
    FiniteCoherentConfiguration,
    PointLabel,
    RelationId,
)

PointPair = tuple[PointLabel, PointLabel]


class CoherentConfigurationAnalyzeRequest(StrictModel):
    """Analyze one complete bounded ordered-pair relation partition."""

    configuration: CoherentConfigurationInput


class Fiber(StrictModel):
    """One diagonal fibre, bound to its diagonal relation."""

    relation_id: RelationId
    points: tuple[PointLabel, ...] = Field(min_length=1)


class TransposeRelation(StrictModel):
    """The declared relation equal to the transpose of one source relation."""

    relation_id: RelationId
    transpose_relation_id: RelationId


class IntersectionNumber(StrictModel):
    """One exact coefficient ``p_ij^k`` in the relation basis."""

    left_relation_id: RelationId
    right_relation_id: RelationId
    target_relation_id: RelationId
    value: int = Field(ge=0)


class DiagonalRelationMixedObstruction(StrictModel):
    kind: Literal["DIAGONAL_RELATION_MIXED"] = "DIAGONAL_RELATION_MIXED"
    relation_id: RelationId
    first_pair: PointPair
    second_pair: PointPair


class TransposeRelationMismatchObstruction(StrictModel):
    kind: Literal["TRANSPOSE_RELATION_MISMATCH"] = "TRANSPOSE_RELATION_MISMATCH"
    relation_id: RelationId
    transpose_relation_id: RelationId
    first_pair: PointPair


class NonconstantIntersectionNumberObstruction(StrictModel):
    kind: Literal["NONCONSTANT_INTERSECTION_NUMBER"] = "NONCONSTANT_INTERSECTION_NUMBER"
    left_relation_id: RelationId
    right_relation_id: RelationId
    target_relation_id: RelationId
    first_pair: PointPair
    second_pair: PointPair
    first_count: int = Field(ge=0)
    second_count: int = Field(ge=0)


CoherenceObstruction = Annotated[
    DiagonalRelationMixedObstruction
    | TransposeRelationMismatchObstruction
    | NonconstantIntersectionNumberObstruction,
    Field(discriminator="kind"),
]


class CoherentConfigurationAnalyzeResult(StrictModel):
    """One structural exact coherence conclusion and its declared derivation."""

    configuration: CoherentConfigurationInput
    status: Literal["COHERENT_CONFIGURATION", "NOT_COHERENT"]
    coherent_configuration: FiniteCoherentConfiguration | None = None
    fibers: tuple[Fiber, ...] = Field(max_length=MAX_COHERENT_CONFIGURATION_POINTS)
    transpose_map: tuple[TransposeRelation, ...] = Field(
        max_length=MAX_COHERENT_CONFIGURATION_RELATIONS
    )
    intersection_numbers: tuple[IntersectionNumber, ...] = Field(
        max_length=MAX_COHERENT_CONFIGURATION_RELATIONS**3
    )
    obstruction: CoherenceObstruction | None = None
    method: Literal["DIRECT_COMPLETE_PAIR_PARTITION_REPLAY"] = (
        "DIRECT_COMPLETE_PAIR_PARTITION_REPLAY"
    )

    @model_validator(mode="after")
    def require_result_shape(self) -> Self:
        """Check bounded internal consistency without replaying the kernel."""

        relation_ids = set(self.configuration.relation_ids)
        points = set(self.configuration.points)
        relation_count = len(relation_ids)
        if self.status == "NOT_COHERENT":
            if (
                self.coherent_configuration is not None
                or self.fibers
                or self.transpose_map
                or self.intersection_numbers
                or self.obstruction is None
            ):
                raise PydanticCustomError(
                    "coherent_configuration.result_shape",
                    "a non-coherent result must carry exactly one obstruction",
                )
            return self
        if (
            self.coherent_configuration is None
            or self.coherent_configuration.model_dump()
            != self.configuration.model_dump()
            or self.obstruction is not None
        ):
            raise PydanticCustomError(
                "coherent_configuration.result_shape",
                "a coherent result must carry its source configuration and no obstruction",
            )
        if (
            len(self.transpose_map) != relation_count
            or len(self.intersection_numbers) != relation_count**3
        ):
            raise PydanticCustomError(
                "coherent_configuration.result_shape",
                "a coherent result must contain its complete relation-derived shapes",
            )
        if {entry.relation_id for entry in self.transpose_map} != relation_ids or any(
            entry.transpose_relation_id not in relation_ids
            for entry in self.transpose_map
        ):
            raise PydanticCustomError(
                "coherent_configuration.result_transpose_map",
                "transpose-map relation identifiers must belong to the source",
            )
        intersection_keys = {
            (entry.left_relation_id, entry.right_relation_id, entry.target_relation_id)
            for entry in self.intersection_numbers
        }
        if len(intersection_keys) != relation_count**3 or any(
            set(key) - relation_ids for key in intersection_keys
        ):
            raise PydanticCustomError(
                "coherent_configuration.result_intersection_numbers",
                "intersection-number relation triples must be the source cube",
            )
        if len({entry.relation_id for entry in self.fibers}) != len(self.fibers) or any(
            entry.relation_id not in relation_ids
            or not set(entry.points) <= points
            or tuple(sorted(entry.points)) != entry.points
            for entry in self.fibers
        ):
            raise PydanticCustomError(
                "coherent_configuration.result_fibers",
                "fiber data must use sorted source points and relations",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        configuration: CoherentConfigurationInput,
        status: Literal["COHERENT_CONFIGURATION", "NOT_COHERENT"],
        coherent_configuration: FiniteCoherentConfiguration | None,
        fibers: tuple[Fiber, ...],
        transpose_map: tuple[TransposeRelation, ...],
        intersection_numbers: tuple[IntersectionNumber, ...],
        obstruction: CoherenceObstruction | None,
    ) -> Self:
        """Construct one fresh result after the owner-local kernel completed."""

        return cls.model_validate(
            {
                "configuration": configuration,
                "status": status,
                "coherent_configuration": coherent_configuration,
                "fibers": fibers,
                "transpose_map": transpose_map,
                "intersection_numbers": intersection_numbers,
                "obstruction": obstruction,
            }
        )


__all__ = [
    "CoherenceObstruction",
    "CoherentConfigurationAnalyzeRequest",
    "CoherentConfigurationAnalyzeResult",
    "DiagonalRelationMixedObstruction",
    "Fiber",
    "IntersectionNumber",
    "NonconstantIntersectionNumberObstruction",
    "TransposeRelation",
    "TransposeRelationMismatchObstruction",
]
