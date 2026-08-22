"""Domain-owned finite group operations."""

from __future__ import annotations

from jacobian.canonical import format_canonical_integer
from jacobian.math.group import (
    conjugacy_classes,
    element_order,
    group_orbit,
    group_order,
    subgroup_lattice,
)
from jacobian.math.group._models import (
    GroupConjugacyClassesRequest,
    GroupConjugacyClassesResult,
    GroupElementOrderRequest,
    GroupElementOrderResult,
    GroupOrbitRequest,
    GroupOrbitResult,
    GroupOrderResult,
    GroupSubgroupLatticeRequest,
    GroupSubgroupLatticeResult,
    PermutationGroupRequest,
)


def compute_group_order(request: PermutationGroupRequest) -> GroupOrderResult:
    order = group_order(request.degree, [list(g) for g in request.generators])
    return GroupOrderResult(order=format_canonical_integer(order))


def compute_element_order(request: GroupElementOrderRequest) -> GroupElementOrderResult:
    order = element_order(request.degree, list(request.generator))
    return GroupElementOrderResult(order=format_canonical_integer(order))


def compute_group_orbit(request: GroupOrbitRequest) -> GroupOrbitResult:
    orbit = group_orbit(
        request.degree,
        [list(g) for g in request.generators],
        request.point,
    )
    return GroupOrbitResult(orbit=tuple(orbit), point=request.point)


def compute_conjugacy_classes(
    request: GroupConjugacyClassesRequest,
) -> GroupConjugacyClassesResult:
    """Compute the conjugacy classes of a permutation group."""
    from jacobian.math.group._models import (
        ConjugacyClass,
        GroupConjugacyClassesResult,
    )

    classes = conjugacy_classes(request.degree, [list(g) for g in request.generators])
    class_entries = tuple(
        ConjugacyClass(
            elements=tuple(tuple(e) for e, _ in [(elem, 0) for elem in cl[0]]),
            size=cl[1],
        )
        for cl in classes
    )
    return GroupConjugacyClassesResult(
        classes=class_entries,
        class_count=len(class_entries),
    )


def compute_subgroup_lattice(
    request: GroupSubgroupLatticeRequest,
) -> GroupSubgroupLatticeResult:
    """Enumerate all subgroups of a bounded permutation group."""
    from jacobian.math.group._models import (
        GroupSubgroupLatticeResult,
        SubgroupEntry,
    )

    subgroups = subgroup_lattice(request.degree, [list(g) for g in request.generators])
    entries = tuple(
        SubgroupEntry(
            generators=tuple(tuple(g) for g in sg[0]),
            order=sg[1],
        )
        for sg in subgroups
    )
    return GroupSubgroupLatticeResult(
        subgroups=entries,
        subgroup_count=len(entries),
    )
