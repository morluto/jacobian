"""Domain adapter for root isolation operations."""

from __future__ import annotations

from jacobian.contracts.root_isolation import (
    AlgebraicCompareRequest,
    AlgebraicCompareResult,
    RootIsolationResult,
    UnivariatePolynomialRequest,
)
from jacobian.math.root_isolation import compare_algebraic, isolate_real_roots


def compute_root_isolation(request: UnivariatePolynomialRequest) -> RootIsolationResult:
    roots = isolate_real_roots(
        [{"num": c.num, "den": c.den} for c in request.coefficients_descending]
    )
    return RootIsolationResult(
        roots=(),
        multiplicities=tuple(m for _, m in roots),
    )


def compute_algebraic_compare(
    request: AlgebraicCompareRequest,
) -> AlgebraicCompareResult:
    order = compare_algebraic(
        [{"num": c.num, "den": c.den} for c in request.left.polynomial],
        request.left.isolating_interval_lower,
        request.left.isolating_interval_upper,
        [{"num": c.num, "den": c.den} for c in request.right.polynomial],
        request.right.isolating_interval_lower,
        request.right.isolating_interval_upper,
    )
    return AlgebraicCompareResult(order=order)
