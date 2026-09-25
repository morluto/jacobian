"""Bounded quotients of finite simplicial-set prefixes by congruences."""

from __future__ import annotations

import json
from typing import NoReturn

from jacobian._execution import request_checkpoint
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology.simplicial_sets._models import (
    MAX_SIMPLICIAL_SET_DEGREE,
    FiniteTruncatedSimplicialSet,
)
from jacobian.math.topology.simplicial_sets._models import (
    FiniteTruncatedSimplicialSet as _CanonicalSimplicialSet,
)
from jacobian.math.topology.simplicial_sets.maps import TruncatedSimplicialMap
from jacobian.math.topology.simplicial_sets.operations import admit_tables, from_tables
from jacobian.math.topology.simplicial_sets.quotient_models import (
    MAX_CLASS_ID,
    SimplicialSetQuotientRequest,
    SimplicialSetQuotientResult,
)

MAX_QUOTIENT_CHECK_WORK = 11_000
MAX_QUOTIENT_RESULT_BYTES = 40_000
_QUOTIENT_OUTPUT_OVERHEAD = 4_096


def _invalid(reason: str, message: str, *location: str | int) -> NoReturn:
    raise OperationDomainValidationError(
        location=location or ("degree_class_ids",),
        code=f"simplicial_set.quotient.{reason}",
        message=message,
    )


def _identity_work(max_degree: int, sizes: tuple[int, ...]) -> int:
    face_face = sum(
        degree * (degree + 1) // 2 * sizes[degree]
        for degree in range(2, max_degree + 1)
    )
    degeneracy_degeneracy = sum(
        (degree + 1) * (degree + 2) // 2 * sizes[degree]
        for degree in range(max_degree - 1)
    )
    mixed = sum(
        (degree + 1) * (degree + 2) * sizes[degree] for degree in range(max_degree)
    )
    # Each identity compares two composite rows on a source degree axis.
    return 2 * (face_face + degeneracy_degeneracy + mixed)


def _map_cell_count(max_degree: int, sizes: tuple[int, ...]) -> int:
    return sum(
        (degree + 1) * sizes[degree] for degree in range(1, max_degree + 1)
    ) + sum((degree + 1) * sizes[degree] for degree in range(max_degree))


def _preflight(
    request: SimplicialSetQuotientRequest,
) -> FiniteTruncatedSimplicialSet:
    if type(request) is not SimplicialSetQuotientRequest:
        _invalid(
            "request_type", "request must be a typed simplicial-set quotient request"
        )
    try:
        source = request.simplicial_set
        class_id_rows = request.degree_class_ids
    except (AttributeError, TypeError) as exc:
        _invalid("request_invalid", f"request fields are malformed: {exc}")
    if type(source) is not FiniteTruncatedSimplicialSet:
        _invalid("source_type", "source must be a finite truncated simplicial set")
    try:
        max_degree = source.max_degree
        total_simplices = source.total_simplices
        checked_identities = source.checked_identities
        sets = source.sets
        face_maps = source.face_maps
        degeneracy_maps = source.degeneracy_maps
    except AttributeError as exc:
        _invalid("source_invalid", f"source fields are malformed: {exc}")
    if type(max_degree) is not int or not 0 <= max_degree <= MAX_SIMPLICIAL_SET_DEGREE:
        _invalid("source_degree", "source degree is outside the finite prefix bound")
    if type(checked_identities) is not int or checked_identities < 0:
        _invalid("source_invalid", "source identity count is malformed")
    # Structural admission runs before any serialization walk so a native
    # caller cannot make json.dumps traverse an unbounded label or map row that
    # the shared label/axis path would reject. This is cheaper than the identity
    # replay and owns every structural check, including label length bounds.
    sizes = admit_tables(max_degree, sets, face_maps, degeneracy_maps)
    if type(total_simplices) is not int or total_simplices != sum(sizes):
        _invalid("source_axes", "source total does not match its degree sizes")
    if type(class_id_rows) is not tuple or len(class_id_rows) != max_degree + 1:
        _invalid("degree_coverage", "class IDs must cover each source degree")
    for degree, class_ids in enumerate(class_id_rows):
        if (
            type(class_ids) is not tuple
            or len(class_ids) != sizes[degree]
            or any(
                type(class_id) is not int or not 0 <= class_id <= MAX_CLASS_ID
                for class_id in class_ids
            )
        ):
            _invalid(
                "class_axis_invalid",
                f"degree {degree} class IDs must be JSON-safe nonnegative integers "
                "on the source axis",
                "degree_class_ids",
                degree,
            )

    identity_work = _identity_work(max_degree, sizes)
    map_cells = _map_cell_count(max_degree, sizes)
    # Source and quotient identity replay, congruence scans, table construction,
    # and the projection rows are all linear in these admitted finite tables.
    work = 2 * identity_work + 3 * map_cells + sum(sizes)
    if work > MAX_QUOTIENT_CHECK_WORK:
        raise OperationResourceAdmissionError(
            location=("simplicial_set",),
            code="simplicial_set.quotient.work_budget_exceeded",
            message=(
                f"quotient congruence work {work} exceeds "
                f"{MAX_QUOTIENT_CHECK_WORK} admitted table steps"
            ),
        )

    try:
        source_bytes = len(
            json.dumps(
                source.model_dump(mode="json"),
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
    except (TypeError, ValueError, AttributeError) as exc:
        _invalid("source_invalid", f"source fields cannot be serialized: {exc}")
    max_target_labels = 6 * sum(sizes)
    target_map_rows = sum(degree + 1 for degree in range(1, max_degree + 1)) + sum(
        degree + 1 for degree in range(max_degree)
    )
    target_tables = max_target_labels + 3 * map_cells + 2 * target_map_rows
    projection = 3 * sum(sizes) + 2 * (max_degree + 1)
    output_bytes = source_bytes + target_tables + projection + _QUOTIENT_OUTPUT_OVERHEAD
    if output_bytes > MAX_QUOTIENT_RESULT_BYTES:
        raise OperationResourceAdmissionError(
            location=("simplicial_set",),
            code="simplicial_set.quotient.output_budget_exceeded",
            message=(
                f"estimated quotient output {output_bytes} bytes exceeds "
                f"{MAX_QUOTIENT_RESULT_BYTES}"
            ),
        )
    return source


def simplicial_set_quotient(
    request: SimplicialSetQuotientRequest,
) -> SimplicialSetQuotientResult:
    """Form the exact degreewise quotient when the proposed relation is a congruence."""
    source = _preflight(request)
    class_id_rows = request.degree_class_ids
    request_checkpoint("before simplicial quotient source identity replay")
    source_check = from_tables(
        source.max_degree,
        source.sets,
        source.face_maps,
        source.degeneracy_maps,
    )
    if source_check.status != "SIMPLICIAL_SET" or source_check.simplicial_set is None:
        _invalid(
            "source_invalid",
            "source tables do not satisfy every visible simplicial identity",
            "simplicial_set",
        )
    source = source_check.simplicial_set

    class_positions: list[tuple[int, ...]] = []
    representatives: list[tuple[int, ...]] = []
    quotient_sets: list[tuple[str, ...]] = []
    for class_ids in class_id_rows:
        positions: dict[int, int] = {}
        degree_representatives: list[int] = []
        row: list[int] = []
        for simplex_index, class_id in enumerate(class_ids):
            quotient_index = positions.get(class_id)
            if quotient_index is None:
                quotient_index = len(positions)
                positions[class_id] = quotient_index
                degree_representatives.append(simplex_index)
            row.append(quotient_index)
        class_positions.append(tuple(row))
        representatives.append(tuple(degree_representatives))
        quotient_sets.append(tuple(f"q{index}" for index in range(len(positions))))

    quotient_faces: list[tuple[tuple[int, ...], ...]] = []
    for degree in range(1, source.max_degree + 1):
        maps = []
        for face_index, face_row in enumerate(source.face_maps[degree - 1]):
            request_checkpoint("during simplicial quotient face congruence check")
            representative_images = tuple(
                class_positions[degree - 1][face_row[representative]]
                for representative in representatives[degree]
            )
            for simplex_index, quotient_index in enumerate(class_positions[degree]):
                if (
                    class_positions[degree - 1][face_row[simplex_index]]
                    != representative_images[quotient_index]
                ):
                    _invalid(
                        "face_congruence_failed",
                        f"equivalent simplices have different d_{face_index} classes",
                        "degree_class_ids",
                        degree,
                        simplex_index,
                    )
            maps.append(representative_images)
        quotient_faces.append(tuple(maps))

    quotient_degeneracies: list[tuple[tuple[int, ...], ...]] = []
    for degree in range(source.max_degree):
        maps = []
        for degeneracy_index, degeneracy_row in enumerate(
            source.degeneracy_maps[degree]
        ):
            request_checkpoint("during simplicial quotient degeneracy congruence check")
            representative_images = tuple(
                class_positions[degree + 1][degeneracy_row[representative]]
                for representative in representatives[degree]
            )
            for simplex_index, quotient_index in enumerate(class_positions[degree]):
                if (
                    class_positions[degree + 1][degeneracy_row[simplex_index]]
                    != representative_images[quotient_index]
                ):
                    _invalid(
                        "degeneracy_congruence_failed",
                        "equivalent simplices have different "
                        f"s_{degeneracy_index} classes",
                        "degree_class_ids",
                        degree,
                        simplex_index,
                    )
            maps.append(representative_images)
        quotient_degeneracies.append(tuple(maps))

    target_sizes = tuple(map(len, quotient_sets))
    target = _CanonicalSimplicialSet._from_kernel(
        max_degree=source.max_degree,
        sets=tuple(quotient_sets),
        face_maps=tuple(quotient_faces),
        degeneracy_maps=tuple(quotient_degeneracies),
        total_simplices=sum(target_sizes),
        checked_identities=source.checked_identities,
    )
    projection = TruncatedSimplicialMap(
        source=source,
        target=target,
        maps=tuple(class_positions),
    )
    return SimplicialSetQuotientResult(quotient_map=projection)


__all__ = [
    "MAX_QUOTIENT_CHECK_WORK",
    "MAX_QUOTIENT_RESULT_BYTES",
    "simplicial_set_quotient",
]
