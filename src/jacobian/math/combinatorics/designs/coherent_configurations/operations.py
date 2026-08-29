"""Exact native analysis of complete finite pair partitions."""

from __future__ import annotations

from dataclasses import dataclass

from jacobian.catalog.models import OperationDomainValidationError

from ._bounds import CoherentConfigurationAdmissionError, require_analysis_admission
from ._models import (
    CoherenceObstruction,
    CoherentConfigurationAnalyzeResult,
    DiagonalRelationMixedObstruction,
    Fiber,
    IntersectionNumber,
    NonconstantIntersectionNumberObstruction,
    TransposeRelation,
    TransposeRelationMismatchObstruction,
)
from .values import (
    CoherentConfigurationInput,
    FiniteCoherentConfiguration,
)


@dataclass(frozen=True)
class FiberData:
    relation_id: str
    points: tuple[str, ...]


@dataclass(frozen=True)
class TransposeData:
    relation_id: str
    transpose_relation_id: str


@dataclass(frozen=True)
class IntersectionData:
    left_relation_id: str
    right_relation_id: str
    target_relation_id: str
    value: int


@dataclass(frozen=True)
class ObstructionData:
    kind: str
    relation_id: str | None = None
    transpose_relation_id: str | None = None
    left_relation_id: str | None = None
    right_relation_id: str | None = None
    target_relation_id: str | None = None
    first_pair: tuple[str, str] | None = None
    second_pair: tuple[str, str] | None = None
    first_count: int | None = None
    second_count: int | None = None


@dataclass(frozen=True)
class AnalysisData:
    fibres: tuple[FiberData, ...] = ()
    transpose_map: tuple[TransposeData, ...] = ()
    intersection_numbers: tuple[IntersectionData, ...] = ()
    obstruction: ObstructionData | None = None


def _relation_cells(
    source: CoherentConfigurationInput,
) -> dict[str, tuple[tuple[int, int], ...]]:
    cells: dict[str, list[tuple[int, int]]] = {
        relation_id: [] for relation_id in source.relation_ids
    }
    for left, row in enumerate(source.relation_matrix):
        for right, relation_id in enumerate(row):
            cells[relation_id].append((left, right))
    return {relation_id: tuple(value) for relation_id, value in cells.items()}


def _fiber_data(
    source: CoherentConfigurationInput,
    cells: dict[str, tuple[tuple[int, int], ...]],
) -> tuple[tuple[FiberData, ...], ObstructionData | None]:
    fibres: list[FiberData] = []
    for relation_id in source.relation_ids:
        relation_cells = cells[relation_id]
        diagonal = tuple(left for left, right in relation_cells if left == right)
        if not diagonal:
            continue
        off_diagonal = next(
            ((left, right) for left, right in relation_cells if left != right), None
        )
        if off_diagonal is not None:
            return (), ObstructionData(
                kind="DIAGONAL_RELATION_MIXED",
                relation_id=relation_id,
                first_pair=(source.points[diagonal[0]], source.points[diagonal[0]]),
                second_pair=(
                    source.points[off_diagonal[0]],
                    source.points[off_diagonal[1]],
                ),
            )
        fibres.append(
            FiberData(
                relation_id=relation_id,
                points=tuple(source.points[index] for index in diagonal),
            )
        )
    return tuple(fibres), None


def _transpose_data(
    source: CoherentConfigurationInput,
    cells: dict[str, tuple[tuple[int, int], ...]],
) -> tuple[tuple[TransposeData, ...], ObstructionData | None]:
    transpose_map: list[TransposeData] = []
    for relation_id in source.relation_ids:
        relation_cells = cells[relation_id]
        first_left, first_right = relation_cells[0]
        partner = source.relation_matrix[first_right][first_left]
        source_transpose = {(right, left) for left, right in relation_cells}
        partner_cells = set(cells[partner])
        if source_transpose != partner_cells:
            witness = min(source_transpose.symmetric_difference(partner_cells))
            return (), ObstructionData(
                kind="TRANSPOSE_RELATION_MISMATCH",
                relation_id=relation_id,
                transpose_relation_id=partner,
                first_pair=(source.points[witness[0]], source.points[witness[1]]),
            )
        transpose_map.append(
            TransposeData(
                relation_id=relation_id,
                transpose_relation_id=partner,
            )
        )
    return tuple(transpose_map), None


def _intersection_data(
    source: CoherentConfigurationInput,
    cells: dict[str, tuple[tuple[int, int], ...]],
) -> tuple[tuple[IntersectionData, ...], ObstructionData | None]:
    matrix = source.relation_matrix
    points = source.points
    point_count = len(points)
    entries: list[IntersectionData] = []
    for left_relation_id in source.relation_ids:
        for right_relation_id in source.relation_ids:
            for target_relation_id in source.relation_ids:
                target_cells = cells[target_relation_id]
                first_left, first_right = target_cells[0]
                first_count = sum(
                    matrix[first_left][middle] == left_relation_id
                    and matrix[middle][first_right] == right_relation_id
                    for middle in range(point_count)
                )
                for left, right in target_cells[1:]:
                    count = sum(
                        matrix[left][middle] == left_relation_id
                        and matrix[middle][right] == right_relation_id
                        for middle in range(point_count)
                    )
                    if count != first_count:
                        return (), ObstructionData(
                            kind="NONCONSTANT_INTERSECTION_NUMBER",
                            left_relation_id=left_relation_id,
                            right_relation_id=right_relation_id,
                            target_relation_id=target_relation_id,
                            first_pair=(points[first_left], points[first_right]),
                            second_pair=(points[left], points[right]),
                            first_count=first_count,
                            second_count=count,
                        )
                entries.append(
                    IntersectionData(
                        left_relation_id=left_relation_id,
                        right_relation_id=right_relation_id,
                        target_relation_id=target_relation_id,
                        value=first_count,
                    )
                )
    return tuple(entries), None


def _analyze(source: CoherentConfigurationInput) -> AnalysisData:
    """Return complete exact derived data or the first failed coherence axiom."""

    require_analysis_admission(source)
    cells = _relation_cells(source)
    fibres, obstruction = _fiber_data(source, cells)
    if obstruction is not None:
        return AnalysisData(obstruction=obstruction)
    transpose_map, obstruction = _transpose_data(source, cells)
    if obstruction is not None:
        return AnalysisData(obstruction=obstruction)
    intersection_numbers, obstruction = _intersection_data(source, cells)
    if obstruction is not None:
        return AnalysisData(obstruction=obstruction)
    return AnalysisData(
        fibres=fibres,
        transpose_map=transpose_map,
        intersection_numbers=intersection_numbers,
    )


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


def _computed_analysis_result(
    source: CoherentConfigurationInput, data: AnalysisData
) -> CoherentConfigurationAnalyzeResult:
    """Build a fresh result from one native analysis pass."""

    if data.obstruction is not None:
        return CoherentConfigurationAnalyzeResult._from_kernel(
            configuration=source,
            status="NOT_COHERENT",
            coherent_configuration=None,
            fibers=(),
            transpose_map=(),
            intersection_numbers=(),
            obstruction=_obstruction_model(data.obstruction),
        )
    return CoherentConfigurationAnalyzeResult._from_kernel(
        configuration=source,
        status="COHERENT_CONFIGURATION",
        coherent_configuration=FiniteCoherentConfiguration._from_kernel(source),
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


def analyze_configuration(
    configuration: CoherentConfigurationInput,
) -> CoherentConfigurationAnalyzeResult:
    """Analyze one complete relation partition with one native kernel pass."""

    try:
        data = _analyze(configuration)
    except CoherentConfigurationAdmissionError as exc:
        raise OperationDomainValidationError(
            location=("configuration",),
            code="coherent_configuration.analysis_not_admitted",
            message=str(exc),
        ) from exc
    return _computed_analysis_result(configuration, data)


__all__ = ["analyze_configuration"]
