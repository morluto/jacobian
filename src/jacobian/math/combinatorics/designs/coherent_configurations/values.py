"""Canonical values for bounded finite coherent configurations.

A coherent configuration is not a generic relation language.  Its sole
authoritative relation datum is a complete colouring of one finite labelled
ordered-pair carrier.  All fibres, transpose partners, and intersection
numbers are derived from that matrix.
"""

from __future__ import annotations

from typing import Annotated, Self
from unicodedata import is_normalized

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel

MAX_COHERENT_CONFIGURATION_POINTS = 12
MAX_COHERENT_CONFIGURATION_RELATIONS = 16
MAX_POINT_LABEL_BYTES = 64
MAX_RELATION_ID_BYTES = 32
MAX_COHERENT_CONFIGURATION_SOURCE_BYTES = 16_384
MAX_COHERENT_CONFIGURATION_RESULT_BYTES = 1_048_576
MAX_INTERSECTION_LOOKUPS_PER_PASS = (
    MAX_COHERENT_CONFIGURATION_RELATIONS**2 * MAX_COHERENT_CONFIGURATION_POINTS**3
)
MAX_ANALYSIS_WORK = 4 * MAX_INTERSECTION_LOOKUPS_PER_PASS

PointLabel = Annotated[str, Field(min_length=1, max_length=MAX_POINT_LABEL_BYTES)]
RelationId = Annotated[str, Field(min_length=1, max_length=MAX_RELATION_ID_BYTES)]


class CoherentConfigurationInput(StrictModel):
    """A bounded complete colouring of ``X x X`` awaiting coherence analysis."""

    points: tuple[PointLabel, ...] = Field(
        min_length=1, max_length=MAX_COHERENT_CONFIGURATION_POINTS
    )
    relation_ids: tuple[RelationId, ...] = Field(
        min_length=1, max_length=MAX_COHERENT_CONFIGURATION_RELATIONS
    )
    relation_matrix: tuple[tuple[RelationId, ...], ...] = Field(
        min_length=1, max_length=MAX_COHERENT_CONFIGURATION_POINTS
    )

    @model_validator(mode="after")
    def require_complete_bounded_pair_partition(self) -> Self:
        if any(
            len(point.encode("utf-8")) > MAX_POINT_LABEL_BYTES for point in self.points
        ):
            raise PydanticCustomError(
                "coherent_configuration.point_label_bytes",
                f"point labels must not exceed {MAX_POINT_LABEL_BYTES} UTF-8 bytes",
            )
        if tuple(sorted(self.points)) != self.points or len(set(self.points)) != len(
            self.points
        ):
            raise PydanticCustomError(
                "coherent_configuration.points_canonical",
                "points must be unique and sorted",
            )
        if any(not is_normalized("NFC", point) for point in self.points):
            raise PydanticCustomError(
                "coherent_configuration.points_nfc", "points must use Unicode NFC"
            )
        if tuple(sorted(self.relation_ids)) != self.relation_ids or len(
            set(self.relation_ids)
        ) != len(self.relation_ids):
            raise PydanticCustomError(
                "coherent_configuration.relation_ids_canonical",
                "relation_ids must be unique and sorted",
            )
        if any(
            not is_normalized("NFC", relation_id) for relation_id in self.relation_ids
        ):
            raise PydanticCustomError(
                "coherent_configuration.relation_ids_nfc",
                "relation_ids must use Unicode NFC",
            )
        if any(
            len(relation_id.encode("utf-8")) > MAX_RELATION_ID_BYTES
            for relation_id in self.relation_ids
        ):
            raise PydanticCustomError(
                "coherent_configuration.relation_id_bytes",
                f"relation_ids must not exceed {MAX_RELATION_ID_BYTES} UTF-8 bytes",
            )
        if len(self.relation_ids) > len(self.points) ** 2:
            raise PydanticCustomError(
                "coherent_configuration.relation_count",
                "relation count cannot exceed ordered-pair cells",
            )
        if len(self.relation_matrix) != len(self.points) or any(
            len(row) != len(self.points) for row in self.relation_matrix
        ):
            raise PydanticCustomError(
                "coherent_configuration.matrix_square",
                "relation_matrix must be square on points",
            )
        declared = set(self.relation_ids)
        if any(
            relation_id not in declared
            for row in self.relation_matrix
            for relation_id in row
        ):
            raise PydanticCustomError(
                "coherent_configuration.matrix_relation_ids",
                "relation_matrix must use only declared relation_ids",
            )
        used = {relation_id for row in self.relation_matrix for relation_id in row}
        if used != declared:
            raise PydanticCustomError(
                "coherent_configuration.relation_ids_used",
                "every declared relation_id must occur in relation_matrix",
            )

        return self


class FiniteCoherentConfiguration(CoherentConfigurationInput):
    """A kernel-established coherent configuration with canonical pair data."""

    @classmethod
    def _from_kernel(cls, source: CoherentConfigurationInput) -> Self:
        """Construct a value after the exact owner-local coherence kernel passed."""

        return cls.model_construct(
            points=source.points,
            relation_ids=source.relation_ids,
            relation_matrix=source.relation_matrix,
        )


__all__ = [
    "MAX_ANALYSIS_WORK",
    "MAX_COHERENT_CONFIGURATION_POINTS",
    "MAX_COHERENT_CONFIGURATION_RELATIONS",
    "MAX_COHERENT_CONFIGURATION_RESULT_BYTES",
    "MAX_COHERENT_CONFIGURATION_SOURCE_BYTES",
    "MAX_INTERSECTION_LOOKUPS_PER_PASS",
    "CoherentConfigurationInput",
    "FiniteCoherentConfiguration",
    "PointLabel",
    "RelationId",
]
