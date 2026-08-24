"""Exact native analysis of complete finite pair partitions."""

from __future__ import annotations

import json
from dataclasses import dataclass

from .values import (
    MAX_ANALYSIS_WORK,
    MAX_COHERENT_CONFIGURATION_RESULT_BYTES,
    MAX_COHERENT_CONFIGURATION_SOURCE_BYTES,
    CoherentConfigurationInput,
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


def _source_bytes(source: CoherentConfigurationInput) -> int:
    return len(source.model_dump_json().encode("utf-8"))


def _json_string_byte_bound(value: str) -> int:
    """Conservatively bound one JSON string scalar, including escapes."""

    return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))


def _estimate_result_bytes(source: CoherentConfigurationInput) -> int:
    """Bound the full wire result before materializing its cubic tensor.

    Each tensor entry contains three relation identifiers and fixed JSON/model
    structure.  The fixed allowance is intentionally above its current wire
    shape, so schema-local changes fail admission rather than silently growing
    the public result envelope.
    """

    source_bytes = _source_bytes(source)
    relation_id_bytes = max(
        _json_string_byte_bound(value) for value in source.relation_ids
    )
    point_bytes = max(_json_string_byte_bound(value) for value in source.points)
    relation_count = len(source.relation_ids)
    point_count = len(source.points)
    tensor_entry_bytes = 3 * relation_id_bytes + 128
    fibre_bytes = point_count * (point_bytes + 48)
    transpose_bytes = relation_count * (2 * relation_id_bytes + 64)
    return (
        2 * source_bytes
        + relation_count**3 * tensor_entry_bytes
        + fibre_bytes
        + transpose_bytes
        + 2_048
    )


def _require_analysis_admission(source: CoherentConfigurationInput) -> None:
    """Reject source or predicted replay/result work before cubic expansion."""

    source_bytes = _source_bytes(source)
    if source_bytes > MAX_COHERENT_CONFIGURATION_SOURCE_BYTES:
        raise ValueError("coherent-configuration source exceeds the byte budget")
    point_count = len(source.points)
    relation_count = len(source.relation_ids)
    work = 4 * relation_count**2 * point_count**3
    if work > MAX_ANALYSIS_WORK:
        raise ValueError("coherent-configuration analysis exceeds the work budget")
    if _estimate_result_bytes(source) > MAX_COHERENT_CONFIGURATION_RESULT_BYTES:
        raise ValueError("coherent-configuration result exceeds the byte budget")


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

    _require_analysis_admission(source)
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
