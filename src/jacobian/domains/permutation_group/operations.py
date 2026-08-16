"""Domain adapter for permutation group operations."""
from __future__ import annotations
from jacobian.canonical import format_canonical_integer
from jacobian.contracts.permutation_group import (
    PermutationGroupOrderResult, PermutationGroupOrbitRequest, PermutationGroupOrbitResult,
    PermutationGroupRequest,
)
from jacobian.math.permutation_group import pg_order, pg_orbit

def compute_pg_order(request: PermutationGroupRequest) -> PermutationGroupOrderResult:
    order = pg_order(request.degree, [list(g) for g in request.generators])
    return PermutationGroupOrderResult(order=format_canonical_integer(order))

def compute_pg_orbit(request: PermutationGroupOrbitRequest) -> PermutationGroupOrbitResult:
    orbit = pg_orbit(request.degree, [list(g) for g in request.generators], request.point)
    return PermutationGroupOrbitResult(orbit=tuple(orbit))
