"""Exact bounded levelwise coproducts of finite simplicial-set prefixes."""

from __future__ import annotations

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology.simplicial_sets._models import (
    MAX_SIMPLICES_PER_DEGREE,
    MAX_TOTAL_SIMPLICES,
    FiniteTruncatedSimplicialSet,
)
from jacobian.math.topology.simplicial_sets.coproduct_models import (
    SimplicialSetCoproductRequest,
    SimplicialSetCoproductResult,
    TaggedSimplex,
)
from jacobian.math.topology.simplicial_sets.maps import TruncatedSimplicialMap
from jacobian.math.topology.simplicial_sets.operations import from_tables

MAX_COPRODUCT_MAP_ROWS = 50_000
MAX_COPRODUCT_IDENTITY_WORK = 100_000
MAX_COPRODUCT_OUTPUT_CELLS = 1_000_000


def _identity_work(max_degree: int, sizes: tuple[int, ...]) -> int:
    return (
        sum(sizes[n] * ((n + 1) * n // 2) for n in range(2, max_degree + 1))
        + sum(sizes[n] * ((n + 1) * (n + 2) // 2) for n in range(max_degree - 1))
        + sum(sizes[n] * ((n + 1) * (n + 2)) for n in range(max_degree))
    )


def _require_simplicial_factor(
    factor: FiniteTruncatedSimplicialSet, *, location: str
) -> FiniteTruncatedSimplicialSet:
    checked = from_tables(
        factor.max_degree,
        factor.sets,
        factor.face_maps,
        factor.degeneracy_maps,
    )
    if checked.status != "SIMPLICIAL_SET":
        obstruction = checked.obstruction
        detail = (
            f"{obstruction.identity_family} fails in degree {obstruction.degree}"
            if obstruction is not None
            else "a simplicial identity fails"
        )
        raise OperationDomainValidationError(
            location=(location,),
            code="simplicial_set.coproduct_factor_invalid",
            message=f"{location} factor is not a simplicial set: {detail}",
        )
    assert checked.simplicial_set is not None
    return checked.simplicial_set


def simplicial_set_coproduct(
    request: SimplicialSetCoproductRequest,
) -> SimplicialSetCoproductResult:
    """Construct the finite degreewise disjoint union X ⊔ Y."""
    left, right = request.left, request.right
    left = _require_simplicial_factor(left, location="left")
    right = _require_simplicial_factor(right, location="right")
    if left.max_degree != right.max_degree:
        raise OperationDomainValidationError(
            location=("right", "max_degree"),
            code="simplicial_set.coproduct_degree_mismatch",
            message="coproduct factors must have the same retained maximum degree",
        )

    factor_work = _identity_work(left.max_degree, tuple(map(len, left.sets))) + (
        _identity_work(right.max_degree, tuple(map(len, right.sets)))
    )
    if factor_work > MAX_COPRODUCT_IDENTITY_WORK:
        raise OperationResourceAdmissionError(
            location=("left", "right"),
            code="simplicial_set.coproduct_identity_work_exceeded",
            message=(
                f"factor identity work {factor_work} exceeds "
                f"{MAX_COPRODUCT_IDENTITY_WORK}"
            ),
        )

    sizes = tuple(
        len(left.sets[degree]) + len(right.sets[degree])
        for degree in range(left.max_degree + 1)
    )
    total = sum(sizes)
    if any(size > MAX_SIMPLICES_PER_DEGREE for size in sizes) or total > (
        MAX_TOTAL_SIMPLICES
    ):
        raise OperationResourceAdmissionError(
            location=("left", "right"),
            code="simplicial_set.coproduct_simplex_budget_exceeded",
            message=(
                "degreewise coproduct exceeds the finite simplicial-set carrier "
                f"({MAX_SIMPLICES_PER_DEGREE} per degree, {MAX_TOTAL_SIMPLICES} total)"
            ),
        )

    # A face/degeneracy row contains one image per simplex in its source level.
    map_rows = sum(
        sizes[n] * ((n + 1 if n > 0 else 0) + (n + 1 if n < left.max_degree else 0))
        for n in range(left.max_degree + 1)
    )
    if map_rows > MAX_COPRODUCT_MAP_ROWS:
        raise OperationResourceAdmissionError(
            location=("left", "right"),
            code="simplicial_set.coproduct_map_row_budget_exceeded",
            message=(
                f"coproduct requires {map_rows} map rows, above "
                f"{MAX_COPRODUCT_MAP_ROWS}"
            ),
        )
    # Includes tagged labels, exact tagged axes, inclusions, and all map rows.
    output_cells = total + 2 * map_rows
    if output_cells > MAX_COPRODUCT_OUTPUT_CELLS:
        raise OperationResourceAdmissionError(
            location=("left", "right"),
            code="simplicial_set.coproduct_output_budget_exceeded",
            message=(
                f"estimated coproduct output {output_cells} cells exceeds "
                f"{MAX_COPRODUCT_OUTPUT_CELLS}"
            ),
        )

    labels = tuple(
        tuple(
            [f"L{i}" for i in range(len(left.sets[degree]))]
            + [f"R{i}" for i in range(len(right.sets[degree]))]
        )
        for degree in range(left.max_degree + 1)
    )
    axes: tuple[tuple[TaggedSimplex, ...], ...] = tuple(
        tuple(
            [("left", i) for i in range(len(left.sets[degree]))]
            + [("right", i) for i in range(len(right.sets[degree]))]
        )
        for degree in range(left.max_degree + 1)
    )

    face_maps = []
    for degree in range(1, left.max_degree + 1):
        offset = len(left.sets[degree - 1])
        level = []
        for face_index in range(degree + 1):
            left_row = left.face_maps[degree - 1][face_index]
            right_row = right.face_maps[degree - 1][face_index]
            level.append(tuple(left_row) + tuple(offset + image for image in right_row))
        face_maps.append(tuple(level))

    degeneracy_maps = []
    for degree in range(left.max_degree):
        offset = len(left.sets[degree + 1])
        level = []
        for degeneracy_index in range(degree + 1):
            left_row = left.degeneracy_maps[degree][degeneracy_index]
            right_row = right.degeneracy_maps[degree][degeneracy_index]
            level.append(tuple(left_row) + tuple(offset + image for image in right_row))
        degeneracy_maps.append(tuple(level))

    checked_identities = (
        sum((n + 1) * n // 2 for n in range(2, left.max_degree + 1))
        + sum((n + 1) * (n + 2) // 2 for n in range(left.max_degree - 1))
        + sum((n + 1) * (n + 2) for n in range(left.max_degree))
    )
    coproduct = FiniteTruncatedSimplicialSet._from_kernel(
        max_degree=left.max_degree,
        sets=labels,
        face_maps=tuple(face_maps),
        degeneracy_maps=tuple(degeneracy_maps),
        total_simplices=total,
        checked_identities=checked_identities,
    )
    left_inclusion = TruncatedSimplicialMap(
        source=left,
        target=coproduct,
        maps=tuple(
            tuple(range(len(left.sets[degree])))
            for degree in range(left.max_degree + 1)
        ),
    )
    right_inclusion = TruncatedSimplicialMap(
        source=right,
        target=coproduct,
        maps=tuple(
            tuple(
                len(left.sets[degree]) + index
                for index in range(len(right.sets[degree]))
            )
            for degree in range(left.max_degree + 1)
        ),
    )
    return SimplicialSetCoproductResult(
        left=left,
        right=right,
        simplicial_set=coproduct,
        tagged_axes=axes,
        left_inclusion=left_inclusion,
        right_inclusion=right_inclusion,
    )


__all__ = ["simplicial_set_coproduct"]
