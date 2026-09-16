"""Exact finite truncated simplicial-set kernel."""

from __future__ import annotations

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology.simplicial_sets._models import (
    MAX_SIMPLICES_PER_DEGREE,
    MAX_SIMPLICIAL_SET_DEGREE,
    MAX_TOTAL_SIMPLICES,
    FiniteTruncatedSimplicialSet,
    SimplicialIdentityObstruction,
    SimplicialSetTablesRequest,
    SimplicialSetTablesResult,
)

DegreeSets = tuple[tuple[str, ...], ...]
MapTable = tuple[tuple[tuple[int, ...], ...], ...]

__all__ = ["admit_tables", "from_tables"]


def admit_tables(
    max_degree: int,
    sets: DegreeSets,
    face_maps: MapTable,
    degeneracy_maps: MapTable,
) -> tuple[int, ...]:
    """Shared native+catalog admission; return admitted degree sizes."""
    if not isinstance(max_degree, int) or not 0 <= max_degree <= (
        MAX_SIMPLICIAL_SET_DEGREE
    ):
        raise OperationDomainValidationError(
            location=("max_degree",),
            code="simplicial_set.degree_out_of_bounds",
            message=f"max_degree must be an integer in 0..{MAX_SIMPLICIAL_SET_DEGREE}",
        )
    if not isinstance(sets, tuple) or len(sets) != max_degree + 1:
        raise OperationDomainValidationError(
            location=("sets",),
            code="simplicial_set.degree_coverage_invalid",
            message=f"sets must cover degrees 0..{max_degree} exactly once",
        )
    sizes: list[int] = []
    for degree, level in enumerate(sets):
        if (
            not isinstance(level, tuple)
            or not 1 <= len(level) <= MAX_SIMPLICES_PER_DEGREE
            or any(not isinstance(label, str) or not label for label in level)
            or len(set(level)) != len(level)
        ):
            raise OperationDomainValidationError(
                location=("sets", degree),
                code="simplicial_set.degree_set_invalid",
                message=f"degree-{degree} labels must be unique nonempty strings",
            )
        sizes.append(len(level))
    if sum(sizes) > MAX_TOTAL_SIMPLICES:
        raise OperationResourceAdmissionError(
            location=("sets",),
            code="simplicial_set.total_simplex_budget_exceeded",
            message=(
                f"{sum(sizes)} simplices exceed the "
                f"{MAX_TOTAL_SIMPLICES}-cell bound"
            ),
        )
    _admit_index_table(face_maps, tuple(sizes), kind="face")
    _admit_index_table(degeneracy_maps, tuple(sizes), kind="degeneracy")
    return tuple(sizes)


def _admit_index_table(
    table: MapTable, sizes: tuple[int, ...], *, kind: str
) -> None:
    expected_levels = len(sizes) - 1
    label = "face_maps" if kind == "face" else "degeneracy_maps"
    if not isinstance(table, tuple) or len(table) != expected_levels:
        raise OperationDomainValidationError(
            location=(label,),
            code=f"simplicial_set.{kind}_table_degree_mismatch",
            message=f"{label} must provide one entry per covered degree",
        )
    for level_index, level in enumerate(table):
        degree = level_index + 1 if kind == "face" else level_index
        target_degree = degree - 1 if kind == "face" else degree + 1
        if not isinstance(level, tuple) or len(level) != degree + 1:
            raise OperationDomainValidationError(
                location=(label, level_index),
                code=f"simplicial_set.{kind}_map_count_mismatch",
                message=f"degree {degree} must carry exactly {degree + 1} maps",
            )
        for map_index, row in enumerate(level):
            if (
                not isinstance(row, tuple)
                or len(row) != sizes[degree]
                or any(
                    not isinstance(target, int) or not 0 <= target < sizes[target_degree]
                    for target in row
                )
            ):
                raise OperationDomainValidationError(
                    location=(label, level_index, map_index),
                    code=f"simplicial_set.{kind}_map_axis_invalid",
                    message=(
                        f"map {map_index} in degree {degree} must send every "
                        f"simplex to a degree-{target_degree} index"
                    ),
                )


def _compose(first: tuple[int, ...], second: tuple[int, ...]) -> tuple[int, ...]:
    """Index row of ``second ∘ first`` for finite index maps."""
    return tuple(second[source] for source in first)


def from_tables(request: SimplicialSetTablesRequest) -> SimplicialSetTablesResult:
    """Check every simplicial identity in degrees <= N for finite tables."""
    sizes = admit_tables(
        request.max_degree, request.sets, request.face_maps, request.degeneracy_maps
    )
    checked = 0
    face_face_obstruction = _check_face_face(request, sizes)
    if face_face_obstruction is not None:
        return SimplicialSetTablesResult._from_kernel(
            status="NOT_A_SIMPLICIAL_SET",
            simplicial_set=None,
            checked_identities=checked + face_face_obstruction[1],
            obstruction=face_face_obstruction[0],
        )
    checked += _face_face_count(request.max_degree)
    degeneracy_obstruction, extra = _check_degeneracy_degeneracy(
        request, sizes, checked
    )
    if degeneracy_obstruction is not None:
        return SimplicialSetTablesResult._from_kernel(
            status="NOT_A_SIMPLICIAL_SET",
            simplicial_set=None,
            checked_identities=checked + extra,
            obstruction=degeneracy_obstruction,
        )
    checked += _degeneracy_degeneracy_count(request.max_degree)
    face_degeneracy_obstruction, extra = _check_face_degeneracy(
        request, sizes, checked
    )
    if face_degeneracy_obstruction is not None:
        return SimplicialSetTablesResult._from_kernel(
            status="NOT_A_SIMPLICIAL_SET",
            simplicial_set=None,
            checked_identities=checked + extra,
            obstruction=face_degeneracy_obstruction,
        )
    checked += _face_degeneracy_count(request.max_degree)
    simplicial_set = FiniteTruncatedSimplicialSet._from_kernel(
        max_degree=request.max_degree,
        sets=request.sets,
        face_maps=request.face_maps,
        degeneracy_maps=request.degeneracy_maps,
        total_simplices=sum(sizes),
        checked_identities=checked,
    )
    return SimplicialSetTablesResult._from_kernel(
        status="SIMPLICIAL_SET",
        simplicial_set=simplicial_set,
        checked_identities=checked,
        obstruction=None,
    )


def _face_face_count(max_degree: int) -> int:
    return sum((degree + 1) * degree // 2 for degree in range(2, max_degree + 1))


def _degeneracy_degeneracy_count(max_degree: int) -> int:
    return sum(
        (degree + 1) * (degree + 2) // 2 for degree in range(max_degree - 1)
    )


def _face_degeneracy_count(max_degree: int) -> int:
    return sum((degree + 1) * (degree + 2) for degree in range(max_degree))


def _first_difference(
    left: tuple[int, ...], right: tuple[int, ...]
) -> int | None:
    for position, (left_value, right_value) in enumerate(zip(left, right)):
        if left_value != right_value:
            return position
    return None


def _check_face_face(
    request: SimplicialSetTablesRequest, sizes: tuple[int, ...]
) -> tuple[SimplicialIdentityObstruction, int] | None:
    position = 0
    for degree in range(2, request.max_degree + 1):
        for outer in range(degree + 1):
            for inner in range(outer):
                left = _compose(
                    request.face_maps[degree - 1][outer],
                    request.face_maps[degree - 2][inner],
                )
                right = _compose(
                    request.face_maps[degree - 1][inner],
                    request.face_maps[degree - 2][outer - 1],
                )
                row = _first_difference(left, right)
                if row is not None:
                    return (
                        SimplicialIdentityObstruction(
                            identity_family="FACE_FACE",
                            degree=degree,
                            left_description=f"d_{inner} d_{outer}",
                            right_description=f"d_{outer - 1} d_{inner}",
                            row=row,
                            left_row=left,
                            right_row=right,
                        ),
                        position,
                    )
                position += 1
    return None


def _check_degeneracy_degeneracy(
    request: SimplicialSetTablesRequest, sizes: tuple[int, ...], checked: int
) -> tuple[SimplicialIdentityObstruction | None, int]:
    del sizes
    position = 0
    for degree in range(request.max_degree - 1):
        for outer in range(degree + 1):
            for inner in range(outer + 1):
                left = _compose(
                    request.degeneracy_maps[degree][outer],
                    request.degeneracy_maps[degree + 1][inner],
                )
                right = _compose(
                    request.degeneracy_maps[degree][inner],
                    request.degeneracy_maps[degree + 1][outer + 1],
                )
                row = _first_difference(left, right)
                if row is not None:
                    return (
                        SimplicialIdentityObstruction(
                            identity_family="DEGENERACY_DEGENERACY",
                            degree=degree,
                            left_description=f"s_{inner} s_{outer}",
                            right_description=f"s_{outer + 1} s_{inner}",
                            row=row,
                            left_row=left,
                            right_row=right,
                        ),
                        position,
                    )
                position += 1
    return None, position


def _check_face_degeneracy(
    request: SimplicialSetTablesRequest, sizes: tuple[int, ...], checked: int
) -> tuple[SimplicialIdentityObstruction | None, int]:
    del sizes, checked
    position = 0
    for degree in range(request.max_degree):
        level_size = len(request.sets[degree])
        for outer in range(degree + 2):
            for inner in range(degree + 1):
                left = _compose(
                    request.degeneracy_maps[degree][inner],
                    request.face_maps[degree][outer],
                )
                if degree == 0 or outer in (inner, inner + 1):
                    right = _identity_row(level_size)
                    right_description = "id"
                elif outer < inner:
                    right = _compose(
                        request.face_maps[degree - 1][outer],
                        request.degeneracy_maps[degree - 1][inner - 1],
                    )
                    right_description = f"s_{inner - 1} d_{outer}"
                else:
                    right = _compose(
                        request.face_maps[degree - 1][outer - 1],
                        request.degeneracy_maps[degree - 1][inner],
                    )
                    right_description = f"s_{inner} d_{outer - 1}"
                row = _first_difference(left, right)
                if row is not None:
                    return (
                        SimplicialIdentityObstruction(
                            identity_family="FACE_DEGENERACY",
                            degree=degree,
                            left_description=f"d_{outer} s_{inner}",
                            right_description=right_description,
                            row=row,
                            left_row=left,
                            right_row=right,
                        ),
                        position,
                    )
                position += 1
    return None, position


def _identity_row(size: int) -> tuple[int, ...]:
    return tuple(range(size))
