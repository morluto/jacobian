"""Catalog adapters for coherent-configuration analysis."""

from __future__ import annotations

from jacobian.math.coherent_configurations._models import (
    CoherenceObstruction,
    CoherentConfigurationAnalyzeRequest,
    CoherentConfigurationAnalyzeResult,
    DiagonalRelationMixedObstruction,
    Fiber,
    IntersectionNumber,
    NonconstantIntersectionNumberObstruction,
    TransposeRelation,
    TransposeRelationMismatchObstruction,
)
from jacobian.math.coherent_configurations.operations import (
    AnalysisData,
    ObstructionData,
    _analyze,
)
from jacobian.math.coherent_configurations.values import (
    CoherentConfigurationInput,
    FiniteCoherentConfiguration,
)

__all__ = ["analysis_models_from_source", "compute_analyze"]


def _obstruction_model(obstruction: ObstructionData) -> CoherenceObstruction:
    if obstruction.kind == "DIAGONAL_RELATION_MIXED":
        assert (
            obstruction.relation_id is not None
            and obstruction.first_pair is not None
            and obstruction.second_pair is not None
        )
        return DiagonalRelationMixedObstruction(
            relation_id=obstruction.relation_id,
            first_pair=obstruction.first_pair,
            second_pair=obstruction.second_pair,
        )
    if obstruction.kind == "TRANSPOSE_RELATION_MISMATCH":
        assert (
            obstruction.relation_id is not None
            and obstruction.transpose_relation_id is not None
            and obstruction.first_pair is not None
        )
        return TransposeRelationMismatchObstruction(
            relation_id=obstruction.relation_id,
            transpose_relation_id=obstruction.transpose_relation_id,
            first_pair=obstruction.first_pair,
        )
    assert obstruction.kind == "NONCONSTANT_INTERSECTION_NUMBER"
    assert (
        obstruction.left_relation_id is not None
        and obstruction.right_relation_id is not None
        and obstruction.target_relation_id is not None
        and obstruction.first_pair is not None
        and obstruction.second_pair is not None
        and obstruction.first_count is not None
        and obstruction.second_count is not None
    )
    return NonconstantIntersectionNumberObstruction(
        left_relation_id=obstruction.left_relation_id,
        right_relation_id=obstruction.right_relation_id,
        target_relation_id=obstruction.target_relation_id,
        first_pair=obstruction.first_pair,
        second_pair=obstruction.second_pair,
        first_count=obstruction.first_count,
        second_count=obstruction.second_count,
    )


def analysis_models_from_source(
    source: CoherentConfigurationInput,
) -> CoherentConfigurationAnalyzeResult:
    """Build the unique source-bound result fields without result validation."""

    data: AnalysisData = _analyze(source)
    if data.obstruction is not None:
        return CoherentConfigurationAnalyzeResult.model_construct(
            configuration=source,
            status="NOT_COHERENT",
            coherent_configuration=None,
            fibers=(),
            transpose_map=(),
            intersection_numbers=(),
            obstruction=_obstruction_model(data.obstruction),
        )
    configuration = FiniteCoherentConfiguration(**source.model_dump())
    return CoherentConfigurationAnalyzeResult.model_construct(
        configuration=source,
        status="COHERENT_CONFIGURATION",
        coherent_configuration=configuration,
        fibers=tuple(
            Fiber(relation_id=fibre.relation_id, points=fibre.points)
            for fibre in data.fibres
        ),
        transpose_map=tuple(
            TransposeRelation(
                relation_id=entry.relation_id,
                transpose_relation_id=entry.transpose_relation_id,
            )
            for entry in data.transpose_map
        ),
        intersection_numbers=tuple(
            IntersectionNumber(
                left_relation_id=entry.left_relation_id,
                right_relation_id=entry.right_relation_id,
                target_relation_id=entry.target_relation_id,
                value=entry.value,
            )
            for entry in data.intersection_numbers
        ),
        obstruction=None,
    )


def compute_analyze(
    request: CoherentConfigurationAnalyzeRequest,
) -> CoherentConfigurationAnalyzeResult:
    """Analyze one complete relation partition and replay the typed result."""

    expected = analysis_models_from_source(request.configuration)
    return CoherentConfigurationAnalyzeResult(
        configuration=expected.configuration,
        status=expected.status,
        coherent_configuration=expected.coherent_configuration,
        fibers=expected.fibers,
        transpose_map=expected.transpose_map,
        intersection_numbers=expected.intersection_numbers,
        obstruction=expected.obstruction,
    )
