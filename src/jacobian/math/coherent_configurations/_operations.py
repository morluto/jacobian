"""Catalog adapters for coherent-configuration analysis."""

from __future__ import annotations

from pydantic_core import PydanticCustomError

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
    MAX_COHERENT_CONFIGURATION_RESULT_BYTES,
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


def _require_trusted_result_shape(result: CoherentConfigurationAnalyzeResult) -> None:
    """Check every non-replay invariant of one fresh kernel result.

    The public result validator separately recomputes the complete coherence
    derivation for independently supplied data.  The producing kernel already
    performed that cubic scan, so its construction verifies only the shape and
    transport invariants that a replay would otherwise hide.
    """

    relation_ids = set(result.configuration.relation_ids)
    points = set(result.configuration.points)
    relation_count = len(relation_ids)

    if result.status == "NOT_COHERENT":
        if (
            result.coherent_configuration is not None
            or result.fibers
            or result.transpose_map
            or result.intersection_numbers
            or result.obstruction is None
        ):
            raise PydanticCustomError(
                "coherent_configuration.trusted_result_shape",
                "a non-coherent result must carry exactly one obstruction",
            )
    else:
        if result.coherent_configuration is None or result.obstruction is not None:
            raise PydanticCustomError(
                "coherent_configuration.trusted_result_shape",
                "a coherent result must carry its configuration and no obstruction",
            )
        if len(result.transpose_map) != relation_count:
            raise PydanticCustomError(
                "coherent_configuration.trusted_result_transpose_map",
                "a coherent result must give one transpose partner per relation",
            )
        if len(result.intersection_numbers) != relation_count**3:
            raise PydanticCustomError(
                "coherent_configuration.trusted_result_intersection_numbers",
                "a coherent result must give one intersection number per relation triple",
            )
        if len({entry.relation_id for entry in result.transpose_map}) != relation_count:
            raise PydanticCustomError(
                "coherent_configuration.trusted_result_transpose_map",
                "transpose-map relation identifiers must be unique",
            )
        if {entry.relation_id for entry in result.transpose_map} != relation_ids or any(
            entry.transpose_relation_id not in relation_ids
            for entry in result.transpose_map
        ):
            raise PydanticCustomError(
                "coherent_configuration.trusted_result_transpose_map",
                "transpose-map relation identifiers must belong to the source",
            )
        intersection_keys = {
            (
                entry.left_relation_id,
                entry.right_relation_id,
                entry.target_relation_id,
            )
            for entry in result.intersection_numbers
        }
        if len(intersection_keys) != relation_count**3 or any(
            set(key) - relation_ids for key in intersection_keys
        ):
            raise PydanticCustomError(
                "coherent_configuration.trusted_result_intersection_numbers",
                "intersection-number relation triples must be the source cube",
            )
        if len({entry.relation_id for entry in result.fibers}) != len(result.fibers):
            raise PydanticCustomError(
                "coherent_configuration.trusted_result_fibers",
                "fiber relation identifiers must be unique",
            )
        if any(
            entry.relation_id not in relation_ids
            or not set(entry.points) <= points
            or tuple(sorted(entry.points)) != entry.points
            for entry in result.fibers
        ):
            raise PydanticCustomError(
                "coherent_configuration.trusted_result_fibers",
                "fiber data must use sorted source points and relations",
            )

    if (
        len(result.model_dump_json().encode("utf-8"))
        > MAX_COHERENT_CONFIGURATION_RESULT_BYTES
    ):
        raise PydanticCustomError(
            "coherent_configuration.result_bytes",
            "coherent-configuration result exceeds the byte budget",
        )


def _computed_analysis_result(
    source: CoherentConfigurationInput, data: AnalysisData
) -> CoherentConfigurationAnalyzeResult:
    """Build a fresh result without replaying the producing cubic analysis."""

    if data.obstruction is not None:
        result = CoherentConfigurationAnalyzeResult.model_construct(
            configuration=source,
            status="NOT_COHERENT",
            coherent_configuration=None,
            fibers=(),
            transpose_map=(),
            intersection_numbers=(),
            obstruction=_obstruction_model(data.obstruction),
        )
    else:
        # ``source`` has already been parsed through CoherentConfigurationInput
        # and ``data`` is the just-computed exact kernel output.  Avoid the
        # FiniteCoherentConfiguration validator's duplicate complete analysis;
        # external payloads still take that validation and the result replay.
        configuration = FiniteCoherentConfiguration.model_construct(
            **source.model_dump()
        )
        result = CoherentConfigurationAnalyzeResult.model_construct(
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
    _require_trusted_result_shape(result)
    return result


def analysis_models_from_source(
    source: CoherentConfigurationInput,
) -> CoherentConfigurationAnalyzeResult:
    """Recompute the canonical result used to validate supplied payloads."""

    return _computed_analysis_result(source, _analyze(source))


def compute_analyze(
    request: CoherentConfigurationAnalyzeRequest,
) -> CoherentConfigurationAnalyzeResult:
    """Analyze one complete relation partition with one charged kernel pass."""

    return _computed_analysis_result(
        request.configuration, _analyze(request.configuration)
    )
