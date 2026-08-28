"""Domain adapter for finite topological space operations."""

from __future__ import annotations

from jacobian.math.topology.finite.spaces._models import (
    BoundaryResult,
    ClosureResult,
    ContinuousCheckRequest,
    ContinuousCheckResult,
    InteriorResult,
    KolmogorovQuotientRequest,
    KolmogorovQuotientResult,
    SubsetRequest,
)
from jacobian.math.topology.finite.spaces.operations import (
    boundary,
    closure,
    continuous_check,
    interior,
    kolmogorov_quotient,
)

__all__ = [
    "compute_boundary",
    "compute_closure",
    "compute_continuous_check",
    "compute_interior",
    "compute_kolmogorov_quotient",
]


def compute_interior(request: SubsetRequest) -> InteriorResult:
    result = interior(request.space, frozenset(request.subset))
    return InteriorResult(interior=tuple(sorted(result)))


def compute_closure(request: SubsetRequest) -> ClosureResult:
    result = closure(request.space, frozenset(request.subset))
    return ClosureResult(closure=tuple(sorted(result)))


def compute_boundary(request: SubsetRequest) -> BoundaryResult:
    result = boundary(request.space, frozenset(request.subset))
    return BoundaryResult(boundary=tuple(sorted(result)))


def compute_continuous_check(
    request: ContinuousCheckRequest,
) -> ContinuousCheckResult:
    result = continuous_check(request.point_map)
    return ContinuousCheckResult(is_continuous=result)


def compute_kolmogorov_quotient(
    request: KolmogorovQuotientRequest,
) -> KolmogorovQuotientResult:
    return kolmogorov_quotient(request.space)
