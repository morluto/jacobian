"""Domain-owned finite group operations."""

from __future__ import annotations

from jacobian.canonical import format_canonical_integer
from jacobian.math.group import (
    element_order,
    group_orbit,
    group_order,
    group_stabilizer,
)
from jacobian.math.group._models import (
    GroupElementOrderRequest,
    GroupElementOrderResult,
    GroupOrbitRequest,
    GroupOrbitResult,
    GroupOrderResult,
    GroupStabilizerRequest,
    GroupStabilizerResult,
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


def compute_group_stabilizer(request: GroupStabilizerRequest) -> GroupStabilizerResult:
    stabilizer_generators = group_stabilizer(
        request.degree,
        [list(g) for g in request.generators],
        request.point,
    )
    return GroupStabilizerResult(
        degree=request.degree,
        point=request.point,
        generators=tuple(tuple(g) for g in stabilizer_generators),
        source_generators=request.generators,
    )
