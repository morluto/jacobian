"""Domain-owned finite group operations."""

from __future__ import annotations

from jacobian.canonical import format_canonical_integer
from jacobian.math.group import (
    element_order,
    group_conjugacy_classes,
    group_orbit,
    group_order,
    group_stabilizer,
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
    GroupStabilizerRequest,
    GroupStabilizerResult,
    GroupSubgroupLatticeRequest,
    GroupSubgroupLatticeResult,
    PermutationGroupRequest,
)


def compute_group_order(request: PermutationGroupRequest) -> GroupOrderResult:
    order = group_order(request)
    return GroupOrderResult(order=format_canonical_integer(order))


def compute_element_order(request: GroupElementOrderRequest) -> GroupElementOrderResult:
    order = element_order(request.degree, list(request.generator))
    return GroupElementOrderResult(order=format_canonical_integer(order))


def compute_group_orbit(request: GroupOrbitRequest) -> GroupOrbitResult:
    orbit = group_orbit(request.group, request.point)
    return GroupOrbitResult(orbit=tuple(orbit), point=request.point)


def compute_group_conjugacy_classes(
    request: GroupConjugacyClassesRequest,
) -> GroupConjugacyClassesResult:
    classes = group_conjugacy_classes(
        request.degree,
        [list(g) for g in request.generators],
    )
    return GroupConjugacyClassesResult(
        classes=tuple(tuple(tuple(p) for p in cls) for cls in classes),
    )


def compute_group_stabilizer(request: GroupStabilizerRequest) -> GroupStabilizerResult:
    return GroupStabilizerResult(
        point=request.point,
        source=request.group,
        stabilizer=group_stabilizer(request.group, request.point),
    )


def compute_subgroup_lattice(
    request: GroupSubgroupLatticeRequest,
) -> GroupSubgroupLatticeResult:
    from jacobian.math.group.operations import SubgroupLatticeBudgetExceededError

    source = PermutationGroupRequest(
        degree=request.degree, generators=request.generators
    )
    try:
        subgroups = subgroup_lattice(source)
    except SubgroupLatticeBudgetExceededError as error:
        return GroupSubgroupLatticeResult(
            outcome="LIMIT_EXCEEDED",
            degree=request.degree,
            generators=request.generators,
            detail=str(error),
        )
    return GroupSubgroupLatticeResult(
        degree=request.degree,
        generators=request.generators,
        subgroups=tuple(subgroups),
        subgroup_count=len(subgroups),
    )
