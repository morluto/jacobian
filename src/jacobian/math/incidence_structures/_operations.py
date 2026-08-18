"""Exact incidence structure operations."""

from jacobian.math.incidence_structures._models import (
    DegreeProfileResult,
    IncidenceMatrixRequest,
    IncidenceMatrixResult,
)


def compute_incidence_matrix(request: IncidenceMatrixRequest) -> IncidenceMatrixResult:
    """Compute the 0/1 incidence matrix."""
    inc = request.incidence
    points = inc.points
    block_ids = inc.block_ids
    blocks = inc.blocks

    matrix = []
    for p in points:
        row = []
        for block in blocks:
            row.append(1 if p in block else 0)
        matrix.append(tuple(row))

    return IncidenceMatrixResult(
        points=points,
        block_ids=block_ids,
        matrix=tuple(matrix),
    )


def compute_degree_profile(request: IncidenceMatrixRequest) -> DegreeProfileResult:
    """Compute per-point and per-block degree profiles."""
    inc = request.incidence
    blocks = inc.blocks

    point_degrees = []
    for p in inc.points:
        deg = sum(1 for block in blocks if p in block)
        point_degrees.append((p, deg))

    block_degrees = []
    for j, block in enumerate(blocks):
        block_degrees.append((inc.block_ids[j], len(block)))

    total = sum(len(block) for block in blocks)

    return DegreeProfileResult(
        point_degrees=tuple(point_degrees),
        block_degrees=tuple(block_degrees),
        total_incidences=total,
    )
