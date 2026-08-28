"""Domain-owned finite group operations."""

from __future__ import annotations

from pydantic_core import PydanticCustomError

from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.groups import (
    element_order,
    group_conjugacy_classes,
    group_orbit,
    group_order,
    group_stabilizer,
    subgroup_lattice,
)
from jacobian.math.groups._models import (
    MAX_CONJUGACY_CLASSES_GROUP_ORDER,
    MAX_SUBGROUP_LATTICE_GROUP_ORDER,
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
    PermutationGroup,
    _require_bounded_group_order,
)


def compute_group_order(request: PermutationGroup) -> GroupOrderResult:
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
    try:
        _require_bounded_group_order(
            request.degree,
            request.generators,
            MAX_CONJUGACY_CLASSES_GROUP_ORDER,
            "conjugacy classes",
        )
    except PydanticCustomError as error:
        raise OperationDomainValidationError(
            location=("generators",), code=error.type, message=str(error)
        ) from error
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


def compute_subgroup_lattice(
    request: GroupSubgroupLatticeRequest,
) -> GroupSubgroupLatticeResult:
    from jacobian.math.groups.operations import SubgroupLatticeBudgetExceededError

    try:
        _require_bounded_group_order(
            request.degree,
            request.generators,
            MAX_SUBGROUP_LATTICE_GROUP_ORDER,
            "subgroup lattice enumeration",
        )
    except PydanticCustomError as error:
        raise OperationDomainValidationError(
            location=("generators",), code=error.type, message=str(error)
        ) from error

    source = PermutationGroup(degree=request.degree, generators=request.generators)
    try:
        subgroups = subgroup_lattice(source)
    except SubgroupLatticeBudgetExceededError as error:
        return GroupSubgroupLatticeResult._limit_exceeded_from_kernel(
            request, str(error)
        )
    return GroupSubgroupLatticeResult._computed_from_kernel(request, tuple(subgroups))
