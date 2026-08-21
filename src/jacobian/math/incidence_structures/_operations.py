"""Exact incidence structure operations."""

from collections.abc import Callable
from itertools import combinations

from jacobian.math.incidence_structures._models import (
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
    IncidenceStructure,
    IntersectionsRequest,
    IntersectionsResult,
    LeviGraphRequest,
    LeviGraphResult,
    RestrictionRequest,
    RestrictionResult,
)


def _point_sort_key(points: tuple[str, ...]) -> Callable[[str], int]:
    """Return a sort key function based on the point ordering."""
    index = {p: i for i, p in enumerate(points)}
    return lambda p: index[p]


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


def compute_containment_profile(
    request: ContainmentProfileRequest,
) -> ContainmentProfileResult:
    """Compute t-subset containment multiplicity profiles."""
    inc = request.incidence
    points = inc.points
    blocks = [set(block) for block in inc.blocks]
    t = request.t

    subsets = list(combinations(points, t))

    counts: dict[tuple[str, ...], int] = {}
    for subset in subsets:
        s_set = set(subset)
        count = sum(1 for block in blocks if s_set <= block)
        counts[subset] = count

    subset_profile = tuple((subset, counts[subset]) for subset in subsets)

    histogram_dict: dict[int, int] = {}
    for count in counts.values():
        histogram_dict[count] = histogram_dict.get(count, 0) + 1

    histogram = tuple(sorted(histogram_dict.items()))
    values = [counts[s] for s in subsets]
    min_mult = min(values) if values else 0
    max_mult = max(values) if values else 0
    is_constant = min_mult == max_mult

    constant_lambda = min_mult if is_constant else None

    return ContainmentProfileResult(
        t=t,
        subset_profile=subset_profile,
        histogram=histogram,
        min_multiplicity=min_mult,
        max_multiplicity=max_mult,
        is_constant=is_constant,
        constant_lambda=constant_lambda,
    )


def compute_intersections(request: IntersectionsRequest) -> IntersectionsResult:
    """Compute block intersection profiles."""
    inc = request.incidence
    block_ids = inc.block_ids
    blocks = inc.blocks
    points = inc.points
    n = len(blocks)

    sort_key = _point_sort_key(points)

    pairwise = []
    histogram_dict: dict[int, int] = {}

    for i in range(n):
        for j in range(i + 1, n):
            inter = set(blocks[i]) & set(blocks[j])
            inter_sorted = tuple(sorted(inter, key=sort_key))
            size = len(inter)
            pairwise.append((block_ids[i], block_ids[j], inter_sorted, size))
            histogram_dict[size] = histogram_dict.get(size, 0) + 1

    histogram = tuple(sorted(histogram_dict.items()))

    return IntersectionsResult(
        pairwise=tuple(pairwise),
        histogram=histogram,
    )


def compute_dual(request: DualRequest) -> DualResult:
    """Compute the dual incidence structure (swap points and blocks)."""
    inc = request.incidence
    points = inc.points
    block_ids = inc.block_ids
    blocks = inc.blocks

    sort_key = _point_sort_key(block_ids)

    dual_points = block_ids
    dual_block_ids = points

    dual_blocks = []
    for original_point in points:
        dual_block = []
        for j, bid in enumerate(block_ids):
            if original_point in blocks[j]:
                dual_block.append(bid)
        dual_blocks.append(tuple(sorted(dual_block, key=sort_key)))

    point_map = tuple((p, p) for p in points)
    block_map = tuple((b, b) for b in block_ids)

    return DualResult(
        incidence=IncidenceStructure(
            points=dual_points,
            block_ids=dual_block_ids,
            blocks=tuple(dual_blocks),
        ),
        points=dual_points,
        block_ids=dual_block_ids,
        blocks=tuple(dual_blocks),
        point_map=point_map,
        block_map=block_map,
    )


def compute_complement(request: ComplementRequest) -> ComplementResult:
    """Compute the complement incidence structure."""
    inc = request.incidence
    points = inc.points
    block_ids = inc.block_ids
    blocks = inc.blocks

    point_set = set(points)
    sort_key = _point_sort_key(points)

    complement_blocks = []
    correspondence = []

    for j, bid in enumerate(block_ids):
        original = blocks[j]
        comp = point_set - set(original)
        comp_sorted = tuple(sorted(comp, key=sort_key))
        complement_blocks.append(comp_sorted)
        correspondence.append((bid, original, comp_sorted))

    return ComplementResult(
        points=points,
        block_ids=block_ids,
        blocks=tuple(complement_blocks),
        correspondence=tuple(correspondence),
    )


def compute_restriction(request: RestrictionRequest) -> RestrictionResult:
    """Restrict to a point subset and/or block subset."""
    inc = request.incidence
    points = inc.points
    block_ids = inc.block_ids
    blocks = inc.blocks

    sort_key = _point_sort_key(points)

    if request.block_ids:
        block_id_to_idx = {bid: i for i, bid in enumerate(block_ids)}
        block_indices = [block_id_to_idx[bid] for bid in request.block_ids]
    else:
        block_indices = list(range(len(block_ids)))

    new_block_ids = [block_ids[i] for i in block_indices]
    new_blocks_sets = [set(blocks[i]) for i in block_indices]

    if request.points:
        point_set = set(request.points)
        new_points = [p for p in points if p in point_set]
        new_blocks_sets = [block & point_set for block in new_blocks_sets]
    else:
        new_points = list(points)

    final_block_ids = []
    final_blocks = []
    for i, bid in enumerate(new_block_ids):
        final_block_ids.append(bid)
        final_blocks.append(tuple(sorted(new_blocks_sets[i], key=sort_key)))

    return RestrictionResult(
        points=tuple(new_points),
        block_ids=tuple(final_block_ids),
        blocks=tuple(final_blocks),
    )


def compute_derived_residual(
    request: DerivedResidualRequest,
) -> DerivedResidualResult:
    """Compute the derived or residual incidence structure at a point.

    The derived structure is on P \\ {p}, formed from blocks containing p,
    with p removed from each selected block.  The residual structure is
    on P \\ {p}, formed from blocks not containing p.
    """
    inc = request.incidence
    points = inc.points
    block_ids = inc.block_ids
    blocks = inc.blocks
    p = request.point
    kind = request.kind

    if p not in points:
        raise ValueError("anchor point must be a declared point")

    new_points = [x for x in points if x != p]
    sort_key = _point_sort_key(points)
    new_block_ids = []
    new_blocks = []
    source_blocks = []

    for j in range(len(blocks)):
        contains_p = p in blocks[j]
        if kind == "derived" and contains_p:
            new_block = tuple(sorted((x for x in blocks[j] if x != p), key=sort_key))
            new_block_ids.append(block_ids[j])
            new_blocks.append(new_block)
            source_blocks.append(block_ids[j])
        elif kind == "residual" and not contains_p:
            new_block_ids.append(block_ids[j])
            new_blocks.append(blocks[j])
            source_blocks.append(block_ids[j])

    if not new_block_ids:
        if kind == "derived":
            raise ValueError(
                "derived structure requires at least one block containing the point"
            )
        raise ValueError(
            "residual structure requires at least one block not containing the point"
        )

    return DerivedResidualResult(
        kind=kind,
        anchor_point=p,
        points=tuple(new_points),
        block_ids=tuple(new_block_ids),
        blocks=tuple(new_blocks),
        source_blocks=tuple(source_blocks),
    )


def compute_levi_graph(request: LeviGraphRequest) -> LeviGraphResult:
    """Compute the Levi graph (bipartite incidence graph)."""
    inc = request.incidence
    points = inc.points
    block_ids = inc.block_ids
    blocks = inc.blocks

    left_vertices = tuple(f"p:{p}" for p in points)
    right_vertices = tuple(f"b:{b}" for b in block_ids)

    edges = []
    for j, block in enumerate(blocks):
        right = f"b:{block_ids[j]}"
        for p in block:
            edges.append((f"p:{p}", right))

    return LeviGraphResult(
        left_vertices=left_vertices,
        right_vertices=right_vertices,
        edges=tuple(edges),
    )


def compute_gram(request: GramRequest) -> GramResult:
    """Compute the Gram / concordance matrix.

    For axis='point', returns N N^T (point-codegree matrix).
    For axis='block', returns N^T N (block-intersection matrix).
    """
    inc = request.incidence
    points = inc.points
    block_ids = inc.block_ids
    blocks = inc.blocks

    n = len(points)
    m = len(blocks)
    matrix = []
    for i in range(n):
        row = []
        for j in range(m):
            row.append(1 if points[i] in blocks[j] else 0)
        matrix.append(row)

    if request.axis == "point":
        labels = points
        gram = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                gram[i][j] = sum(matrix[i][k] * matrix[j][k] for k in range(m))
        result_matrix = tuple(tuple(row) for row in gram)
    else:
        labels = block_ids
        gram = [[0] * m for _ in range(m)]
        for i in range(m):
            for j in range(m):
                gram[i][j] = sum(matrix[k][i] * matrix[k][j] for k in range(n))
        result_matrix = tuple(tuple(row) for row in gram)

    return GramResult(
        axis=request.axis,
        labels=labels,
        matrix=result_matrix,
    )
