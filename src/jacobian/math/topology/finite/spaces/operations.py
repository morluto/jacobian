"""Exact native kernels over finite topological spaces."""

from __future__ import annotations

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)

from ._models import (
    BoundaryResult,
    ClosureResult,
    ContinuousCheckResult,
    InteriorResult,
    KolmogorovQuotientResult,
)
from .values import MAX_POINTS, FiniteTopologicalMap, FiniteTopologicalSpace

__all__ = [
    "boundary",
    "closure",
    "continuous_check",
    "from_preorder",
    "interior",
    "kolmogorov_quotient",
    "minimal_neighbourhoods",
    "specialization_preorder",
    "verify_boundary",
    "verify_closure",
    "verify_continuity",
    "verify_interior",
    "verify_kolmogorov_quotient",
]


MAX_PREORDER_INCIDENCES = 262_144
MAX_PREORDER_BIT_WORK = 268_435_456


def _admit_space(space: FiniteTopologicalSpace) -> None:
    """Bound row storage and bitset work before establishing preorder laws.

    With n points and m retained incidences, constructing n-bit row masks and
    checking each inclusion costs O(n*m) bit operations. The factor eight
    covers mask construction, inclusion, reflexivity and linear owner passes.
    This admits large sparse spaces without charging them for n cubed set work.
    """
    count = len(space.points)
    incidences = sum(map(len, space.preorder))
    if (
        count > MAX_POINTS
        or incidences > MAX_PREORDER_INCIDENCES
        or 8 * count * max(count, incidences) > MAX_PREORDER_BIT_WORK
    ):
        raise OperationResourceAdmissionError(
            location=("space",),
            code="finite_topology_space.preorder_work",
            message="preorder storage or bitset work exceeds the admitted bound",
        )
    rows = tuple(sum(1 << index for index in row) for row in space.preorder)
    for i, row in enumerate(rows):
        if not row & (1 << i):
            raise OperationDomainValidationError(
                location=("space",),
                code="finite_topology_space.preorder_not_reflexive",
                message="preorder must be reflexive",
            )
        if any(rows[j] & row != rows[j] for j in space.preorder[i]):
            raise OperationDomainValidationError(
                location=("space",),
                code="finite_topology_space.preorder_not_transitive",
                message="preorder must be transitive",
            )


def from_preorder(
    points: tuple[str, ...],
    preorder: tuple[tuple[int, ...], ...],
) -> FiniteTopologicalSpace:
    """Construct a finite topological space from a preorder."""
    space = FiniteTopologicalSpace(points=points, preorder=preorder)
    _admit_space(space)
    return space


def specialization_preorder(
    space: FiniteTopologicalSpace,
) -> tuple[tuple[int, ...], ...]:
    """Return the specialization preorder rows."""
    _admit_space(space)
    return space.preorder


def minimal_neighbourhoods(
    space: FiniteTopologicalSpace,
) -> tuple[tuple[int, ...], ...]:
    """Return the minimal open neighbourhood of each point.

    The value stores ``preorder[y] = {x : x <= y}`` (the down-set, i.e. the
    closure of {y}). The minimal open neighbourhood of x is the up-set
    ``{y : x in preorder[y]}``.
    """
    _admit_space(space)
    return _minimal_neighbourhoods(space)


def _minimal_neighbourhoods(
    space: FiniteTopologicalSpace,
) -> tuple[tuple[int, ...], ...]:
    rows: list[list[int]] = [[] for _ in space.points]
    for upper, downset in enumerate(space.preorder):
        for lower in downset:
            rows[lower].append(upper)
    return tuple(tuple(row) for row in rows)


def _admit_subset(space: FiniteTopologicalSpace, subset: frozenset[int]) -> None:
    if any(not 0 <= point < len(space.points) for point in subset):
        raise OperationDomainValidationError(
            location=("subset",),
            code="finite_topology_space.subset_index_range",
            message="subset index out of range",
        )


def interior(space: FiniteTopologicalSpace, subset: frozenset[int]) -> frozenset[int]:
    """Return the interior of a subset (largest open set contained in it)."""
    _admit_space(space)
    _admit_subset(space, subset)
    return _interior(space, subset)


def _interior(space: FiniteTopologicalSpace, subset: frozenset[int]) -> frozenset[int]:
    # X \ closure(X \ A) avoids constructing every open neighbourhood.
    outside = frozenset(range(len(space.points))) - subset
    return frozenset(range(len(space.points))) - _closure(space, outside)


def closure(space: FiniteTopologicalSpace, subset: frozenset[int]) -> frozenset[int]:
    """Return the closure of a subset (smallest closed set containing it)."""
    _admit_space(space)
    _admit_subset(space, subset)
    return _closure(space, subset)


def _closure(space: FiniteTopologicalSpace, subset: frozenset[int]) -> frozenset[int]:
    result: set[int] = set()
    for i in subset:
        result.update(space.preorder[i])
    return frozenset(result)


def boundary(space: FiniteTopologicalSpace, subset: frozenset[int]) -> frozenset[int]:
    """Return the boundary of a subset: closure minus interior."""
    _admit_space(space)
    _admit_subset(space, subset)
    cl = _closure(space, subset)
    inter = _interior(space, subset)
    return frozenset(cl - inter)


def continuous_check(map_: FiniteTopologicalMap) -> bool:
    """Check whether a point map between finite topological spaces is continuous.

    A map f: X -> Y is continuous iff for every y in Y, f^{-1}(open_neighbourhood(y))
    is open in X. In the Alexandrov/preorder representation, this means:
    for every x in X and every y with y <= f(x), we need f^{-1}(y) to contain
    the minimal neighbourhood of x. Equivalently: x' <= x implies f(x') <= f(x).
    """
    src = map_.source
    tgt = map_.target
    _admit_space(src)
    _admit_space(tgt)
    target_rows = tuple(set(row) for row in tgt.preorder)
    for i in range(len(src.points)):
        fi = map_.point_map[i]
        for j in src.preorder[i]:
            if map_.point_map[j] not in target_rows[fi]:
                return False
    return True


def kolmogorov_quotient(space: FiniteTopologicalSpace) -> KolmogorovQuotientResult:
    """Return the T0 (Kolmogorov) quotient: identify points with the same
    minimal open neighbourhood."""
    _admit_space(space)
    nbhd_to_class: dict[tuple[int, ...], list[int]] = {}
    for i, row in enumerate(space.preorder):
        key = tuple(sorted(row))
        nbhd_to_class.setdefault(key, []).append(i)
    classes = list(nbhd_to_class.values())
    class_map: dict[int, int] = {}
    for class_idx, cls in enumerate(classes):
        for idx in cls:
            class_map[idx] = class_idx
    quotient_preorder: list[tuple[int, ...]] = []
    for cls in classes:
        representative = cls[0]
        row_set: set[int] = set()
        for j in space.preorder[representative]:
            row_set.add(class_map[j])
        quotient_preorder.append(tuple(sorted(row_set)))
    target = FiniteTopologicalSpace(
        points=tuple(space.points[cls[0]] for cls in classes),
        preorder=tuple(quotient_preorder),
    )
    return KolmogorovQuotientResult(
        quotient_map=FiniteTopologicalMap(
            source=space,
            target=target,
            point_map=tuple(class_map[index] for index in range(len(space.points))),
        )
    )


def verify_continuity(claim: ContinuousCheckResult) -> bool:
    """Check the monotonicity relation of a retained point map.

    A map f is continuous iff x' <= x implies f(x') <= f(x) in the
    specialization preorders. Both endpoint spaces are admitted here because
    a serialized claim is caller-authored.
    """
    source = claim.point_map.source
    target = claim.point_map.target
    try:
        _admit_space(source)
        _admit_space(target)
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False
    target_rows = tuple(set(row) for row in target.preorder)
    monotone = True
    for i in range(len(source.points)):
        image = claim.point_map.point_map[i]
        for j in source.preorder[i]:
            if claim.point_map.point_map[j] not in target_rows[image]:
                monotone = False
                break
        if not monotone:
            break
    return monotone == claim.is_continuous


def verify_interior(claim: InteriorResult) -> bool:
    """Verify an interior claim against its retained finite space and subset."""
    if claim.subset.space != claim.space or claim.interior.space != claim.space:
        return False
    try:
        _admit_space(claim.space)
        expected = _interior(claim.space, frozenset(claim.subset.indices))
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False
    return tuple(sorted(expected)) == claim.interior.indices


def verify_closure(claim: ClosureResult) -> bool:
    """Verify a closure claim against its retained finite space and subset."""
    if claim.subset.space != claim.space or claim.closure.space != claim.space:
        return False
    try:
        _admit_space(claim.space)
        expected = _closure(claim.space, frozenset(claim.subset.indices))
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False
    return tuple(sorted(expected)) == claim.closure.indices


def verify_boundary(claim: BoundaryResult) -> bool:
    """Verify a boundary claim against its retained finite space and subset."""
    if claim.subset.space != claim.space or claim.boundary.space != claim.space:
        return False
    try:
        _admit_space(claim.space)
        subset = frozenset(claim.subset.indices)
        expected = _closure(claim.space, subset) - _interior(claim.space, subset)
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False
    return tuple(sorted(expected)) == claim.boundary.indices


def verify_kolmogorov_quotient(claim: KolmogorovQuotientResult) -> bool:
    """Check the quotient relation directly without rebuilding the quotient.

    Verifies the class map is a consecutive surjection, target points are
    first source representatives, and the target preorder is induced through
    the class map. Producer construction is not replayed.
    """
    quotient_map = claim.quotient_map
    source = quotient_map.source
    target = quotient_map.target
    class_map = quotient_map.point_map
    try:
        _admit_space(source)
        _admit_space(target)
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False
    count = len(source.points)
    if len(class_map) != count:
        return False
    classes = len(target.points)
    if set(class_map) != set(range(classes)) or any(
        not 0 <= image < classes for image in class_map
    ):
        return False
    representatives = [-1] * classes
    row_classes: dict[tuple[int, ...], int] = {}
    for index, row in enumerate(source.preorder):
        image = class_map[index]
        if row not in row_classes:
            if representatives[image] != -1:
                return False
            row_classes[row] = image
            representatives[image] = index
        if image != row_classes[row]:
            return False
    if tuple(source.points[rep] for rep in representatives) != target.points:
        return False
    for target_index in range(classes):
        expected = sorted(
            {class_map[j] for j in source.preorder[representatives[target_index]]}
        )
        if tuple(expected) != tuple(sorted(target.preorder[target_index])):
            return False
    return True
