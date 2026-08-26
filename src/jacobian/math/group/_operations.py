"""Domain-owned finite group operations."""

from __future__ import annotations

from pydantic_core import PydanticCustomError

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
    _check_orbit_stabilizer,
    _check_stabilizer_permutations,
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
    return GroupStabilizerResult._from_kernel(
        request.point,
        request.group,
        group_stabilizer(request.group, request.point),
    )


def verify_group_stabilizer_result(result: GroupStabilizerResult) -> bool:
    """Check a separately supplied stabilizer claim in the group owner."""

    try:
        _check_stabilizer_permutations(
            result.source.degree,
            result.point,
            result.stabilizer.generators,
            result.source.generators,
        )
        _check_orbit_stabilizer(
            result.source.degree,
            result.point,
            result.stabilizer.generators,
            result.source.generators,
        )
    except PydanticCustomError:
        return False
    return True


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
        return GroupSubgroupLatticeResult._limit_exceeded_from_kernel(
            request, str(error)
        )
    return GroupSubgroupLatticeResult._computed_from_kernel(request, tuple(subgroups))


def verify_group_subgroup_lattice_result(result: GroupSubgroupLatticeResult) -> bool:
    """Replay a separately supplied subgroup-lattice claim in its owner."""

    try:
        request = GroupSubgroupLatticeRequest(
            degree=result.degree, generators=result.generators
        )
        if result.outcome != "COMPUTED" or result.subgroups is None:
            return True
        expected = tuple(
            (entry.group.generators, entry.order)
            for entry in subgroup_lattice(
                PermutationGroupRequest(
                    degree=request.degree, generators=request.generators
                )
            )
        )
    except (PydanticCustomError, ValueError):
        return False
    actual = tuple((entry.group.generators, entry.order) for entry in result.subgroups)
    return actual == expected and result.subgroup_count == len(expected)
