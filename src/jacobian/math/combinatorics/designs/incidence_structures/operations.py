"""Native exact incidence-profile and finite-trade operations."""

from __future__ import annotations

from collections.abc import Callable

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.designs.incidence_structures._kernel import (
    containment_profile_data,
    incidence_trade_data,
)
from jacobian.math.combinatorics.designs.incidence_structures._models import (
    ComplementRequest,
    ComplementResult,
    ContainmentProfileRequest,
    ContainmentProfileResult,
    DegreeProfileResult,
    DerivedResidualRequest,
    DerivedResidualResult,
    DualRequest,
    DualResult,
    GramRequest,
    GramResult,
    IncidenceMatrixRequest,
    IncidenceMatrixResult,
    IncidenceMomentComparison,
    IncidenceStructure,
    IncidenceStructureAdmissionError,
    IncidenceTradeRequest,
    IncidenceTradeResult,
    IntersectionsRequest,
    IntersectionsResult,
    LeviGraphRequest,
    LeviGraphResult,
    RestrictionRequest,
    RestrictionResult,
    _require_containment_profile_admitted,
    _require_incidence_trade_admitted,
)


def containment_profile(
    incidence: IncidenceStructure, order: int
) -> ContainmentProfileResult:
    """Return every fixed-order subset containment multiplicity exactly."""

    if not isinstance(incidence, IncidenceStructure):
        raise TypeError("incidence must be an IncidenceStructure")
    if type(order) is not int:
        raise TypeError("containment-profile order must be an integer")
    _require_containment_profile_admitted(incidence, order)
    return ContainmentProfileResult._from_kernel(
        incidence, order, containment_profile_data(incidence, order)
    )


def check_incidence_trade(
    left: IncidenceStructure, right: IncidenceStructure, max_order: int
) -> IncidenceTradeResult:
    """Compare two indexed block families through a positive subset order."""

    if not isinstance(left, IncidenceStructure) or not isinstance(
        right, IncidenceStructure
    ):
        raise TypeError("trade sides must be IncidenceStructure values")
    if type(max_order) is not int:
        raise TypeError("trade comparison order must be an integer")
    _require_incidence_trade_admitted(left, right, max_order)
    zeroth_difference, comparisons, _positive_moments_equal = incidence_trade_data(
        left, right, max_order
    )
    return IncidenceTradeResult._from_kernel(
        left, right, max_order, zeroth_difference, comparisons
    )


def verify_incidence_moment_comparison(
    comparison: IncidenceMomentComparison,
) -> bool:
    """Verify an externally supplied moment comparison within its admission."""

    try:
        _require_containment_profile_admitted(comparison.left, comparison.order)
        _require_containment_profile_admitted(comparison.right, comparison.order)
    except ValueError:
        return False
    left_profile = containment_profile_data(comparison.left, comparison.order)
    right_profile = containment_profile_data(comparison.right, comparison.order)
    expected_differences = tuple(
        (left_entry[0], left_entry[1], right_entry[1])
        for left_entry, right_entry in zip(
            left_profile[0], right_profile[0], strict=True
        )
        if left_entry[1] != right_entry[1]
    )
    actual_differences = tuple(
        (difference.subset, difference.left_multiplicity, difference.right_multiplicity)
        for difference in comparison.differences
    )
    return (
        comparison.left_total == left_profile[2]
        and comparison.right_total == right_profile[2]
        and actual_differences == expected_differences
    )


__all__ = [
    "check_incidence_trade",
    "compute_complement",
    "compute_containment_profile",
    "compute_degree_profile",
    "compute_derived_residual",
    "compute_dual",
    "compute_gram",
    "compute_incidence_matrix",
    "compute_incidence_trade",
    "compute_intersections",
    "compute_levi_graph",
    "compute_restriction",
    "containment_profile",
    "verify_incidence_moment_comparison",
]


def _point_sort_key(points: tuple[str, ...]) -> Callable[[str], int]:
    """Return a sort key function based on the point ordering."""
    index = {point: position for position, point in enumerate(points)}
    return lambda point: index[point]


def compute_incidence_matrix(request: IncidenceMatrixRequest) -> IncidenceMatrixResult:
    """Compute the exact 0/1 incidence matrix."""
    incidence = request.incidence
    matrix = tuple(
        tuple(int(point in block) for block in incidence.blocks)
        for point in incidence.points
    )
    return IncidenceMatrixResult(
        points=incidence.points,
        block_ids=incidence.block_ids,
        matrix=matrix,
    )


def compute_degree_profile(request: IncidenceMatrixRequest) -> DegreeProfileResult:
    """Compute per-point and per-block degree profiles."""
    incidence = request.incidence
    point_degrees = tuple(
        (point, sum(point in block for block in incidence.blocks))
        for point in incidence.points
    )
    block_degrees = tuple(
        (block_id, len(block))
        for block_id, block in zip(
            incidence.block_ids, incidence.blocks, strict=True
        )
    )
    return DegreeProfileResult(
        point_degrees=point_degrees,
        block_degrees=block_degrees,
        total_incidences=sum(len(block) for block in incidence.blocks),
    )


def compute_containment_profile(
    request: ContainmentProfileRequest,
) -> ContainmentProfileResult:
    """Compute t-subset containment multiplicity profiles."""
    _require_containment_profile_admitted(request.incidence, request.t)
    return ContainmentProfileResult._from_kernel(
        request.incidence,
        request.t,
        containment_profile_data(request.incidence, request.t),
    )


def compute_incidence_trade(request: IncidenceTradeRequest) -> IncidenceTradeResult:
    """Compare two indexed block families through a positive subset order."""
    try:
        _require_incidence_trade_admitted(
            request.left, request.right, request.max_order
        )
    except IncidenceStructureAdmissionError as exc:
        raise OperationDomainValidationError(
            location=("left", "right", "max_order"),
            code=f"incidence_structure.{exc.reason}",
            message=str(exc),
        ) from exc
    zeroth_difference, comparisons, _positive_moments_equal = incidence_trade_data(
        request.left, request.right, request.max_order
    )
    return IncidenceTradeResult._from_kernel(
        request.left,
        request.right,
        request.max_order,
        zeroth_difference,
        comparisons,
    )


def compute_intersections(request: IntersectionsRequest) -> IntersectionsResult:
    """Compute block intersection profiles."""
    incidence = request.incidence
    sort_key = _point_sort_key(incidence.points)
    pairwise: list[tuple[str, str, tuple[str, ...], int]] = []
    histogram: dict[int, int] = {}
    for left_index, left_block in enumerate(incidence.blocks):
        for right_index in range(left_index + 1, len(incidence.blocks)):
            intersection = set(left_block) & set(incidence.blocks[right_index])
            ordered = tuple(sorted(intersection, key=sort_key))
            size = len(intersection)
            pairwise.append(
                (
                    incidence.block_ids[left_index],
                    incidence.block_ids[right_index],
                    ordered,
                    size,
                )
            )
            histogram[size] = histogram.get(size, 0) + 1
    return IntersectionsResult(
        pairwise=tuple(pairwise),
        histogram=tuple(sorted(histogram.items())),
    )


def compute_dual(request: DualRequest) -> DualResult:
    """Compute the dual incidence structure (swap points and blocks)."""
    incidence = request.incidence
    sort_key = _point_sort_key(incidence.block_ids)
    dual_points = incidence.block_ids
    dual_block_ids = incidence.points
    dual_blocks = tuple(
        tuple(
            sorted(
                (
                    block_id
                    for block_id, block in zip(
                        incidence.block_ids, incidence.blocks, strict=True
                    )
                    if point in block
                ),
                key=sort_key,
            )
        )
        for point in incidence.points
    )
    return DualResult(
        incidence=IncidenceStructure(
            points=dual_points,
            block_ids=dual_block_ids,
            blocks=dual_blocks,
        ),
        points=dual_points,
        block_ids=dual_block_ids,
        blocks=dual_blocks,
        point_map=tuple((point, point) for point in incidence.points),
        block_map=tuple((block_id, block_id) for block_id in incidence.block_ids),
    )


def compute_complement(request: ComplementRequest) -> ComplementResult:
    """Compute the complement incidence structure."""
    incidence = request.incidence
    point_set = set(incidence.points)
    sort_key = _point_sort_key(incidence.points)
    complement_blocks = tuple(
        tuple(sorted(point_set - set(block), key=sort_key))
        for block in incidence.blocks
    )
    correspondence = tuple(
        (block_id, original, complement)
        for block_id, original, complement in zip(
            incidence.block_ids, incidence.blocks, complement_blocks, strict=True
        )
    )
    return ComplementResult(
        points=incidence.points,
        block_ids=incidence.block_ids,
        blocks=complement_blocks,
        correspondence=correspondence,
    )


def compute_restriction(request: RestrictionRequest) -> RestrictionResult:
    """Restrict to a point subset and/or block subset."""
    incidence = request.incidence
    sort_key = _point_sort_key(incidence.points)
    selected_block_ids = (
        request.block_ids or incidence.block_ids
    )
    blocks_by_id = dict(zip(incidence.block_ids, incidence.blocks, strict=True))
    selected_blocks = [blocks_by_id[block_id] for block_id in selected_block_ids]
    retained_point_set = set(request.points)
    retained_points = (
        tuple(point for point in incidence.points if point in retained_point_set)
        if request.points
        else incidence.points
    )
    retained_point_set = set(retained_points)
    restricted_blocks = tuple(
        tuple(sorted(set(block) & retained_point_set, key=sort_key))
        for block in selected_blocks
    )
    return RestrictionResult(
        points=tuple(retained_points),
        block_ids=tuple(selected_block_ids),
        blocks=restricted_blocks,
    )


def compute_derived_residual(
    request: DerivedResidualRequest,
) -> DerivedResidualResult:
    """Compute the derived or residual incidence structure at a point."""
    incidence = request.incidence
    point = request.point
    if point not in incidence.points:
        raise OperationDomainValidationError(
            location=("point",),
            code="incidence.derived_point_undeclared",
            message="point must be a declared point in the incidence structure",
        )
    selected = tuple(
        (block_id, block)
        for block_id, block in zip(
            incidence.block_ids, incidence.blocks, strict=True
        )
        if (point in block) == (request.kind == "derived")
    )
    sort_key = _point_sort_key(incidence.points)
    blocks = tuple(
        tuple(sorted((member for member in block if member != point), key=sort_key))
        if request.kind == "derived"
        else block
        for _, block in selected
    )
    block_ids = tuple(block_id for block_id, _ in selected)
    return DerivedResidualResult(
        kind=request.kind,
        anchor_point=point,
        points=tuple(member for member in incidence.points if member != point),
        block_ids=block_ids,
        blocks=blocks,
        source_blocks=block_ids,
    )


def compute_levi_graph(request: LeviGraphRequest) -> LeviGraphResult:
    """Compute the Levi graph (bipartite incidence graph)."""
    incidence = request.incidence
    left_vertices = tuple(f"p:{point}" for point in incidence.points)
    right_vertices = tuple(f"b:{block_id}" for block_id in incidence.block_ids)
    edges = tuple(
        (f"p:{point}", f"b:{block_id}")
        for block_id, block in zip(
            incidence.block_ids, incidence.blocks, strict=True
        )
        for point in block
    )
    return LeviGraphResult(
        left_vertices=left_vertices,
        right_vertices=right_vertices,
        edges=edges,
    )


def compute_gram(request: GramRequest) -> GramResult:
    """Compute the Gram / concordance matrix."""
    incidence = request.incidence
    incidence_matrix = tuple(
        tuple(int(point in block) for block in incidence.blocks)
        for point in incidence.points
    )
    if request.axis == "point":
        labels = incidence.points
        matrix = tuple(
            tuple(
                sum(
                    incidence_matrix[left][column]
                    * incidence_matrix[right][column]
                    for column in range(len(incidence.blocks))
                )
                for right in range(len(incidence.points))
            )
            for left in range(len(incidence.points))
        )
    else:
        labels = incidence.block_ids
        matrix = tuple(
            tuple(
                sum(
                    incidence_matrix[row][left] * incidence_matrix[row][right]
                    for row in range(len(incidence.points))
                )
                for right in range(len(incidence.blocks))
            )
            for left in range(len(incidence.blocks))
        )
    return GramResult(axis=request.axis, labels=labels, matrix=matrix)
