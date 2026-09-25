"""Exact bounded degreewise products of finite simplicial-set prefixes."""

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
from jacobian.math.topology.simplicial_sets.maps import TruncatedSimplicialMap
from jacobian.math.topology.simplicial_sets.product_models import (
    SimplicialSetProductRequest,
    SimplicialSetProductResult,
)

MAX_PRODUCT_MAP_ROWS = 50_000
MAX_PRODUCT_IDENTITY_WORK = 100_000
MAX_PRODUCT_OUTPUT_CELLS = 1_000_000


def simplicial_set_product(
    request: SimplicialSetProductRequest,
) -> SimplicialSetProductResult:
    """Construct X x Y with the canonical left-major index-pair axes."""
    left, right = request.left, request.right
    if left.max_degree != right.max_degree:
        raise OperationDomainValidationError(
            location=("right", "max_degree"),
            code="simplicial_set.product_degree_mismatch",
            message="product factors must have the same retained maximum degree",
        )

    sizes = tuple(
        len(left.sets[degree]) * len(right.sets[degree])
        for degree in range(left.max_degree + 1)
    )
    total = sum(sizes)
    if any(size > MAX_SIMPLICES_PER_DEGREE for size in sizes) or total > (
        MAX_TOTAL_SIMPLICES
    ):
        raise OperationResourceAdmissionError(
            location=("left", "right"),
            code="simplicial_set.product_simplex_budget_exceeded",
            message=(
                "degreewise product exceeds the finite simplicial-set carrier "
                f"({MAX_SIMPLICES_PER_DEGREE} per degree, {MAX_TOTAL_SIMPLICES} total)"
            ),
        )

    # Admit every emitted component map row and a conservative JSON-size bound
    # before allocating the product axes or map tables.
    map_rows = sum(
        sizes[n] * ((n + 1 if n > 0 else 0) + (n + 1 if n < left.max_degree else 0))
        for n in range(left.max_degree + 1)
    )
    if map_rows > MAX_PRODUCT_MAP_ROWS:
        raise OperationResourceAdmissionError(
            location=("left", "right"),
            code="simplicial_set.product_map_row_budget_exceeded",
            message=f"product requires {map_rows} map rows, above {MAX_PRODUCT_MAP_ROWS}",
        )
    identity_instances = (
        sum((n + 1) * n // 2 for n in range(2, left.max_degree + 1))
        + sum((n + 1) * (n + 2) // 2 for n in range(left.max_degree - 1))
        + sum((n + 1) * (n + 2) for n in range(left.max_degree))
    )
    identity_work = identity_instances * max(sizes, default=0)
    if identity_work > MAX_PRODUCT_IDENTITY_WORK:
        raise OperationResourceAdmissionError(
            location=("left", "right"),
            code="simplicial_set.product_identity_work_exceeded",
            message=(
                f"product identity bound {identity_work} exceeds "
                f"{MAX_PRODUCT_IDENTITY_WORK}"
            ),
        )
    output_cells = total + 2 * map_rows
    if output_cells > MAX_PRODUCT_OUTPUT_CELLS:
        raise OperationResourceAdmissionError(
            location=("left", "right"),
            code="simplicial_set.product_output_budget_exceeded",
            message=(
                f"estimated product output {output_cells} cells exceeds "
                f"{MAX_PRODUCT_OUTPUT_CELLS}"
            ),
        )

    axes = tuple(
        tuple(
            (i, j) for i in range(len(left.sets[n])) for j in range(len(right.sets[n]))
        )
        for n in range(left.max_degree + 1)
    )
    labels = tuple(
        tuple(f"p{i}" for i in range(sizes[n])) for n in range(left.max_degree + 1)
    )

    faces = []
    for n in range(1, left.max_degree + 1):
        level = []
        for index in range(n + 1):
            lrow, rrow = left.face_maps[n - 1][index], right.face_maps[n - 1][index]
            level.append(
                tuple(lrow[i] * len(right.sets[n - 1]) + rrow[j] for i, j in axes[n])
            )
        faces.append(tuple(level))

    degeneracies = []
    for n in range(left.max_degree):
        level = []
        for index in range(n + 1):
            lrow = left.degeneracy_maps[n][index]
            rrow = right.degeneracy_maps[n][index]
            level.append(
                tuple(lrow[i] * len(right.sets[n + 1]) + rrow[j] for i, j in axes[n])
            )
        degeneracies.append(tuple(level))

    checked_identities = (
        sum((n + 1) * n // 2 for n in range(2, left.max_degree + 1))
        + sum((n + 1) * (n + 2) // 2 for n in range(left.max_degree - 1))
        + sum((n + 1) * (n + 2) for n in range(left.max_degree))
    )
    product = FiniteTruncatedSimplicialSet._from_kernel(
        max_degree=left.max_degree,
        sets=labels,
        face_maps=tuple(faces),
        degeneracy_maps=tuple(degeneracies),
        total_simplices=total,
        checked_identities=checked_identities,
    )

    left_projection = TruncatedSimplicialMap(
        source=product,
        target=left,
        maps=tuple(tuple(i for i, _ in axis) for axis in axes),
    )
    right_projection = TruncatedSimplicialMap(
        source=product,
        target=right,
        maps=tuple(tuple(j for _, j in axis) for axis in axes),
    )
    return SimplicialSetProductResult._from_kernel(
        left=left,
        right=right,
        simplicial_set=product,
        pair_axes=axes,
        left_projection=left_projection,
        right_projection=right_projection,
    )


__all__ = ["simplicial_set_product"]
