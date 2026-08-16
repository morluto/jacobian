"""Domain adapter for permutation group operations."""

from __future__ import annotations

from jacobian.canonical import format_canonical_integer
from jacobian.contracts.permutation_group import (
    PermutationGroupOrbitRequest,
    PermutationGroupOrbitResult,
    PermutationGroupOrderResult,
    PermutationGroupRequest,
)
from jacobian.math.permutation_group import pg_orbit, pg_order


def compute_pg_order(request: PermutationGroupRequest) -> PermutationGroupOrderResult:
    order = pg_order(request.degree, [list(g) for g in request.generators])  # type: ignore[no-untyped-call]
    return PermutationGroupOrderResult(order=format_canonical_integer(order))


def compute_pg_orbit(
    request: PermutationGroupOrbitRequest,
) -> PermutationGroupOrbitResult:
    orbit = pg_orbit(  # type: ignore[no-untyped-call]
        request.degree, [list(g) for g in request.generators], request.point
    )
    return PermutationGroupOrbitResult(orbit=tuple(orbit))
