"""Wire contracts for exact coherent-configuration analysis."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.coherent_configurations.values import (
    MAX_COHERENT_CONFIGURATION_POINTS,
    MAX_COHERENT_CONFIGURATION_RELATIONS,
    MAX_COHERENT_CONFIGURATION_RESULT_BYTES,
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
    """A source-bound exact coherence conclusion and its complete derivation."""

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
    def bind_complete_analysis_to_source(self) -> Self:
        from jacobian.math.coherent_configurations._operations import (
            analysis_models_from_source,
        )

        expected = analysis_models_from_source(self.configuration)
        if self.status != expected.status:
            raise PydanticCustomError(
                "coherent_configuration.result_status",
                "status does not match the exact coherence analysis",
            )
        if self.coherent_configuration != expected.coherent_configuration:
            raise PydanticCustomError(
                "coherent_configuration.result_value",
                "coherent_configuration is not bound to the source",
            )
        if self.fibers != expected.fibers:
            raise PydanticCustomError(
                "coherent_configuration.result_fibers",
                "fibers are not bound to the source configuration",
            )
        if self.transpose_map != expected.transpose_map:
            raise PydanticCustomError(
                "coherent_configuration.result_transpose_map",
                "transpose_map is not bound to the source configuration",
            )
        if self.intersection_numbers != expected.intersection_numbers:
            raise PydanticCustomError(
                "coherent_configuration.result_intersection_numbers",
                "intersection_numbers are not bound to the source configuration",
            )
        if self.obstruction != expected.obstruction:
            raise PydanticCustomError(
                "coherent_configuration.result_obstruction",
                "obstruction does not match the first failed axiom",
            )
        if (
            len(self.model_dump_json().encode("utf-8"))
            > MAX_COHERENT_CONFIGURATION_RESULT_BYTES
        ):
            raise PydanticCustomError(
                "coherent_configuration.result_bytes",
                "coherent-configuration result exceeds the byte budget",
            )
        return self


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
