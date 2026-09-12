"""Native exact incidence-profile and finite-trade operations."""

from __future__ import annotations

import time
from collections.abc import Callable
from itertools import combinations
from typing import Literal

from jacobian._execution import (
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
    request_execution,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.designs.incidence_structures._kernel import (
    containment_profile_data,
    incidence_trade_data,
)
from jacobian.math.combinatorics.designs.incidence_structures._models import (
    MAX_STEINER_SEARCH_STATES,
    ComplementResult,
    ContainmentProfileResult,
    DegreeProfileResult,
    DerivedResidualResult,
    DualResult,
    GramResult,
    IncidenceMatrixResult,
    IncidenceMomentComparison,
    IncidenceStructure,
    IncidenceStructureAdmissionError,
    IncidenceTradeResult,
    IntersectionsResult,
    LeviGraphResult,
    RestrictionResult,
    SteinerTripleSystemResult,
    _require_containment_profile_admitted,
    _require_incidence_trade_admitted,
    _require_steiner_triple_system_admitted,
)
from jacobian.math.combinatorics.exact_cover import (
    ExactCoverRow,
    GeneralizedExactCoverInstance,
    find_generalized_exact_cover,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)
from jacobian.math.matrices.values import IntegerMatrix


def containment_profile(
    incidence: IncidenceStructure | FiniteHypergraph, order: int
) -> ContainmentProfileResult:
    """Return every fixed-order subset containment multiplicity exactly."""

    execution = current_request_execution()
    if execution is None:
        with request_execution(time.monotonic()):
            return containment_profile(incidence, order)
    deadline = execution.started_at + 60
    if execution.deadline is not None:
        deadline = min(deadline, execution.deadline)
    bind_request_deadline(deadline)
    request_checkpoint("before containment profile admission")
    if not isinstance(incidence, (IncidenceStructure, FiniteHypergraph)):
        raise TypeError("incidence must be an IncidenceStructure or FiniteHypergraph")
    if type(order) is not int:
        raise TypeError("containment-profile order must be an integer")
    try:
        _require_containment_profile_admitted(incidence, order)
    except IncidenceStructureAdmissionError as exc:
        raise OperationDomainValidationError(
            location=("incidence", "order"),
            code=f"incidence_structure.{exc.reason}",
            message=str(exc),
        ) from exc
    request_checkpoint("after containment profile admission")
    return ContainmentProfileResult._from_kernel(
        incidence, order, containment_profile_data(incidence, order)
    )


def construct_steiner_triple_system(
    order: int, search_budget: int = MAX_STEINER_SEARCH_STATES
) -> SteinerTripleSystemResult:
    """Construct one STS(order) using bounded exact cover over point pairs.

    Each candidate triple covers exactly three pair constraints. The canonical
    instance is solved by the maintained generalized exact-cover backend, and
    the returned design is independently checked by replaying all pair
    multiplicities before it crosses the operation boundary.
    """
    if type(order) is not int or type(search_budget) is not int:
        raise TypeError("order and search_budget must be integers")

    execution = current_request_execution()
    if execution is None:
        with request_execution(time.monotonic()):
            return construct_steiner_triple_system(order, search_budget)
    deadline = execution.started_at + 60
    if execution.deadline is not None:
        deadline = min(deadline, execution.deadline)
    bind_request_deadline(deadline)
    request_checkpoint("before Steiner construction admission")
    try:
        _require_steiner_triple_system_admitted(order, search_budget)
    except IncidenceStructureAdmissionError as exc:
        raise OperationDomainValidationError(
            location=("order", "search_budget"),
            code=f"incidence_structure.{exc.reason}",
            message=str(exc),
        ) from exc
    request_checkpoint("after Steiner construction admission")

    points = tuple(range(order))
    pairs = tuple(combinations(points, 2))
    triples = tuple(combinations(points, 3))
    pair_labels = {pair: f"pair:{pair[0]:02d}:{pair[1]:02d}" for pair in pairs}
    triple_by_row_id = {
        f"triple:{triple[0]:02d}:{triple[1]:02d}:{triple[2]:02d}": triple
        for triple in triples
    }
    exact_cover = GeneralizedExactCoverInstance(
        primary_items=tuple(sorted(pair_labels.values())),
        secondary_items=(),
        rows=tuple(
            ExactCoverRow(
                row_id=row_id,
                items=tuple(
                    sorted(pair_labels[pair] for pair in combinations(triple, 2))
                ),
            )
            for row_id, triple in triple_by_row_id.items()
        ),
    )
    cover = find_generalized_exact_cover(exact_cover, search_node_limit=search_budget)
    states = cover.searched_node_count
    if cover.status != "FOUND":
        return SteinerTripleSystemResult(
            status="UNKNOWN" if cover.status == "UNKNOWN" else "NOT_FOUND",
            order=order,
            states_explored=states,
        )

    # Replay the defining incidence axiom independently of the cover search.
    pair_multiplicity = dict.fromkeys(pairs, 0)
    selected_row_ids = cover.selected_row_ids
    if selected_row_ids is None:
        raise RuntimeError("FOUND exact-cover result omitted selected rows")
    chosen = tuple(triple_by_row_id[row_id] for row_id in selected_row_ids)
    for triple in chosen:
        for pair in combinations(triple, 2):
            pair_multiplicity[pair] += 1
    if any(value != 1 for value in pair_multiplicity.values()):
        raise RuntimeError("exact-cover search produced an invalid Steiner system")
    canonical_chosen = tuple(sorted(chosen))
    design = IncidenceStructure(
        points=tuple(f"p{point}" for point in points),
        block_ids=tuple(f"b{index}" for index in range(len(canonical_chosen))),
        blocks=tuple(
            tuple(f"p{point}" for point in triple) for triple in canonical_chosen
        ),
    )
    return SteinerTripleSystemResult(
        status="COMPUTED", order=order, design=design, states_explored=states
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
    try:
        _require_incidence_trade_admitted(left, right, max_order)
    except IncidenceStructureAdmissionError as exc:
        raise OperationDomainValidationError(
            location=("left", "right", "max_order"),
            code=f"incidence_structure.{exc.reason}",
            message=str(exc),
        ) from exc
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
    "complement",
    "construct_steiner_triple_system",
    "containment_profile",
    "degree_profile",
    "derived_residual",
    "dual",
    "gram",
    "incidence_matrix",
    "intersections",
    "levi_graph",
    "restriction",
    "verify_incidence_moment_comparison",
]


def _point_sort_key(points: tuple[str, ...]) -> Callable[[str], int]:
    """Return a sort key function based on the point ordering."""
    index = {point: position for position, point in enumerate(points)}
    return lambda point: index[point]


def incidence_matrix(incidence: IncidenceStructure) -> IncidenceMatrixResult:
    """Compute the exact 0/1 incidence matrix."""
    matrix = IntegerMatrix(
        row_count=len(incidence.points),
        column_count=len(incidence.block_ids),
        entries=tuple(
            tuple(int(point in block) for block in incidence.blocks)
            for point in incidence.points
        ),
    )
    return IncidenceMatrixResult(
        points=incidence.points,
        block_ids=incidence.block_ids,
        matrix=matrix,
    )


def degree_profile(incidence: IncidenceStructure) -> DegreeProfileResult:
    """Compute per-point and per-block degree profiles."""
    point_degrees = tuple(
        (point, sum(point in block for block in incidence.blocks))
        for point in incidence.points
    )
    block_degrees = tuple(
        (block_id, len(block))
        for block_id, block in zip(incidence.block_ids, incidence.blocks, strict=True)
    )
    return DegreeProfileResult(
        point_degrees=point_degrees,
        block_degrees=block_degrees,
        total_incidences=sum(len(block) for block in incidence.blocks),
    )


def intersections(incidence: IncidenceStructure) -> IntersectionsResult:
    """Compute block intersection profiles."""
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


def dual(incidence: IncidenceStructure) -> DualResult:
    """Compute the dual incidence structure (swap points and blocks)."""
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


def complement(incidence: IncidenceStructure) -> ComplementResult:
    """Compute the complement incidence structure."""
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


def restriction(
    incidence: IncidenceStructure,
    points: tuple[str, ...],
    block_ids: tuple[str, ...],
) -> RestrictionResult:
    """Restrict to a point subset and/or block subset."""
    sort_key = _point_sort_key(incidence.points)
    selected_block_ids = block_ids or incidence.block_ids
    blocks_by_id = dict(zip(incidence.block_ids, incidence.blocks, strict=True))
    selected_blocks = [blocks_by_id[block_id] for block_id in selected_block_ids]
    retained_point_set = set(points)
    retained_points = (
        tuple(point for point in incidence.points if point in retained_point_set)
        if points
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


def derived_residual(
    incidence: IncidenceStructure, point: str, kind: Literal["derived", "residual"]
) -> DerivedResidualResult:
    """Compute the derived or residual incidence structure at a point."""
    if point not in incidence.points:
        raise OperationDomainValidationError(
            location=("point",),
            code="incidence.derived_point_undeclared",
            message="point must be a declared point in the incidence structure",
        )
    selected = tuple(
        (block_id, block)
        for block_id, block in zip(incidence.block_ids, incidence.blocks, strict=True)
        if (point in block) == (kind == "derived")
    )
    sort_key = _point_sort_key(incidence.points)
    blocks = tuple(
        tuple(sorted((member for member in block if member != point), key=sort_key))
        if kind == "derived"
        else block
        for _, block in selected
    )
    block_ids = tuple(block_id for block_id, _ in selected)
    return DerivedResidualResult(
        kind=kind,
        anchor_point=point,
        points=tuple(member for member in incidence.points if member != point),
        block_ids=block_ids,
        blocks=blocks,
        source_blocks=block_ids,
    )


def levi_graph(incidence: IncidenceStructure) -> LeviGraphResult:
    """Compute the Levi graph (bipartite incidence graph)."""
    left_vertices = tuple(f"p:{point}" for point in incidence.points)
    right_vertices = tuple(f"b:{block_id}" for block_id in incidence.block_ids)
    edges = tuple(
        (f"p:{point}", f"b:{block_id}")
        for block_id, block in zip(incidence.block_ids, incidence.blocks, strict=True)
        for point in block
    )
    return LeviGraphResult(
        left_vertices=left_vertices,
        right_vertices=right_vertices,
        edges=edges,
    )


def gram(incidence: IncidenceStructure, axis: Literal["point", "block"]) -> GramResult:
    """Compute the Gram / concordance matrix."""
    incidence_matrix = tuple(
        tuple(int(point in block) for block in incidence.blocks)
        for point in incidence.points
    )
    if axis == "point":
        labels = incidence.points
        matrix = tuple(
            tuple(
                sum(
                    incidence_matrix[left][column] * incidence_matrix[right][column]
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
    return GramResult(
        axis=axis,
        labels=labels,
        matrix=IntegerMatrix(
            row_count=len(labels),
            column_count=len(labels),
            entries=tuple(tuple(value for value in row) for row in matrix),
        ),
    )
