"""Exact native kernels over finite topological spaces."""

from __future__ import annotations

from jacobian.catalog.models import OperationDomainValidationError

from ._models import (
    BoundaryResult,
    ClosureResult,
    ContinuousCheckResult,
    InteriorResult,
    KolmogorovQuotientResult,
)
from .values import FiniteTopologicalMap, FiniteTopologicalSpace

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


def _admit_space(space: FiniteTopologicalSpace) -> None:
    """Establish the preorder laws within the 64-point carrier."""
    rows = tuple(set(row) for row in space.preorder)
    for i, row in enumerate(rows):
        if i not in row:
            raise OperationDomainValidationError(
                location=("space",),
                code="finite_topology_space.preorder_not_reflexive",
                message="preorder must be reflexive",
            )
        if any(not rows[j] <= row for j in row):
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
    count = len(space.points)
    return tuple(
        tuple(sorted(y for y in range(count) if x in space.preorder[y]))
        for x in range(count)
    )


def interior(space: FiniteTopologicalSpace, subset: frozenset[int]) -> frozenset[int]:
    """Return the interior of a subset (largest open set contained in it)."""
    _admit_space(space)
    return _interior(space, subset)


def _interior(space: FiniteTopologicalSpace, subset: frozenset[int]) -> frozenset[int]:
    neighbourhoods = _minimal_neighbourhoods(space)
    result: set[int] = set()
    for i in range(len(space.points)):
        if set(neighbourhoods[i]).issubset(subset):
            result.add(i)
    return frozenset(result)


def closure(space: FiniteTopologicalSpace, subset: frozenset[int]) -> frozenset[int]:
    """Return the closure of a subset (smallest closed set containing it)."""
    _admit_space(space)
    return _closure(space, subset)


def _closure(space: FiniteTopologicalSpace, subset: frozenset[int]) -> frozenset[int]:
    result: set[int] = set()
    for i in subset:
        if not 0 <= i < len(space.points):
            raise ValueError("subset index out of range")
        result.update(space.preorder[i])
    return frozenset(result)


def boundary(space: FiniteTopologicalSpace, subset: frozenset[int]) -> frozenset[int]:
    """Return the boundary of a subset: closure minus interior."""
    _admit_space(space)
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
    for i in range(len(src.points)):
        fi = map_.point_map[i]
        for j in src.preorder[i]:
            if map_.point_map[j] not in tgt.preorder[fi]:
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
    except OperationDomainValidationError:
        return False
    monotone = True
    for i in range(len(source.points)):
        image = claim.point_map.point_map[i]
        for j in source.preorder[i]:
            if claim.point_map.point_map[j] not in target.preorder[image]:
                monotone = False
                break
        if not monotone:
            break
    return monotone == claim.is_continuous


def verify_interior(claim: InteriorResult) -> bool:
    """Verify an interior claim against its retained finite space and subset."""
    try:
        _admit_space(claim.space)
        expected = _interior(claim.space, frozenset(claim.subset.indices))
    except (OperationDomainValidationError, ValueError, TypeError):
        return False
    return (
        claim.subset.space == claim.space
        and claim.interior.space == claim.space
        and tuple(sorted(expected)) == claim.interior.indices
    )


def verify_closure(claim: ClosureResult) -> bool:
    """Verify a closure claim against its retained finite space and subset."""
    try:
        _admit_space(claim.space)
        expected = _closure(claim.space, frozenset(claim.subset.indices))
    except (OperationDomainValidationError, ValueError, TypeError):
        return False
    return (
        claim.subset.space == claim.space
        and claim.closure.space == claim.space
        and tuple(sorted(expected)) == claim.closure.indices
    )


def verify_boundary(claim: BoundaryResult) -> bool:
    """Verify a boundary claim against its retained finite space and subset."""
    try:
        _admit_space(claim.space)
        subset = frozenset(claim.subset.indices)
        expected = _closure(claim.space, subset) - _interior(claim.space, subset)
    except (OperationDomainValidationError, ValueError, TypeError):
        return False
    return (
        claim.subset.space == claim.space
        and claim.boundary.space == claim.space
        and tuple(sorted(expected)) == claim.boundary.indices
    )


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
    representatives = [
        next(i for i in range(count) if class_map[i] == a) for a in range(classes)
    ]
    if tuple(source.points[rep] for rep in representatives) != target.points:
        return False
    if any(
        (class_map[left] == class_map[right])
        != (source.preorder[left] == source.preorder[right])
        for left in range(count)
        for right in range(left + 1, count)
    ):
        return False
    for target_index in range(classes):
        expected = sorted(
            {class_map[j] for j in source.preorder[representatives[target_index]]}
        )
        if tuple(expected) != tuple(sorted(target.preorder[target_index])):
            return False
    return True
