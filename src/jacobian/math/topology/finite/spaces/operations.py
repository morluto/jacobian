"""Exact native kernels over finite topological spaces."""

from __future__ import annotations

from ._models import KolmogorovQuotientResult
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
]


def from_preorder(
    points: tuple[str, ...],
    preorder: tuple[tuple[int, ...], ...],
) -> FiniteTopologicalSpace:
    """Construct a finite topological space from a preorder."""
    return FiniteTopologicalSpace(points=points, preorder=preorder)


def specialization_preorder(
    space: FiniteTopologicalSpace,
) -> tuple[tuple[int, ...], ...]:
    """Return the specialization preorder rows."""
    return space.preorder


def minimal_neighbourhoods(
    space: FiniteTopologicalSpace,
) -> tuple[tuple[int, ...], ...]:
    """Return the minimal open neighbourhood of each point.

    In an Alexandrov space, the minimal open neighbourhood of x is {y : y <= x}
    = the down-set of x in the specialization preorder.
    """
    return space.preorder


def interior(space: FiniteTopologicalSpace, subset: frozenset[int]) -> frozenset[int]:
    """Return the interior of a subset (largest open set contained in it)."""
    result: set[int] = set()
    for i in range(len(space.points)):
        nbhd = set(space.preorder[i])
        if nbhd.issubset(subset):
            result.add(i)
    return frozenset(result)


def closure(space: FiniteTopologicalSpace, subset: frozenset[int]) -> frozenset[int]:
    """Return the closure of a subset (smallest closed set containing it)."""
    result: set[int] = set()
    for i in subset:
        if not 0 <= i < len(space.points):
            raise ValueError("subset index out of range")
        for j in range(len(space.points)):
            if i in space.preorder[j]:
                result.add(j)
    return frozenset(result)


def boundary(space: FiniteTopologicalSpace, subset: frozenset[int]) -> frozenset[int]:
    """Return the boundary of a subset: closure minus interior."""
    cl = closure(space, subset)
    inter = interior(space, subset)
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
    for i in range(len(src.points)):
        fi = map_.point_map[i]
        for j in src.preorder[i]:
            if map_.point_map[j] not in tgt.preorder[fi]:
                return False
    return True


def kolmogorov_quotient(space: FiniteTopologicalSpace) -> KolmogorovQuotientResult:
    """Return the T0 (Kolmogorov) quotient: identify points with the same
    minimal open neighbourhood."""
    nbhd_to_class: dict[tuple[int, ...], list[int]] = {}
    for i, row in enumerate(space.preorder):
        key = tuple(sorted(row))
        nbhd_to_class.setdefault(key, []).append(i)
    classes = list(nbhd_to_class.values())
    quotient_points = tuple(
        tuple(space.points[idx] for idx in sorted(cls)) for cls in classes
    )
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
    return KolmogorovQuotientResult(
        quotient_points=quotient_points,
        quotient_preorder=tuple(quotient_preorder),
        class_map=tuple(class_map[index] for index in range(len(space.points))),
    )
