"""Wire contracts for exact coherent-configuration analysis."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

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
            raise ValueError("status does not match the exact coherence analysis")
        if self.coherent_configuration != expected.coherent_configuration:
            raise ValueError("coherent_configuration is not bound to the source")
        if self.fibers != expected.fibers:
            raise ValueError("fibers are not bound to the source configuration")
        if self.transpose_map != expected.transpose_map:
            raise ValueError("transpose_map is not bound to the source configuration")
        if self.intersection_numbers != expected.intersection_numbers:
            raise ValueError(
                "intersection_numbers are not bound to the source configuration"
            )
        if self.obstruction != expected.obstruction:
            raise ValueError("obstruction does not match the first failed axiom")
        if (
            len(self.model_dump_json().encode("utf-8"))
            > MAX_COHERENT_CONFIGURATION_RESULT_BYTES
        ):
            raise ValueError("coherent-configuration result exceeds the byte budget")
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
