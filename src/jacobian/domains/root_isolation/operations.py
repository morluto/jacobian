"""Domain adapter for root isolation operations."""

from __future__ import annotations

import sympy

from jacobian.canonical import format_canonical_integer
from jacobian.contracts.exact import CanonicalRational
from jacobian.contracts.root_isolation import (
    AlgebraicCompareRequest,
    AlgebraicCompareResult,
    RootIsolationResult,
    UnivariatePolynomialRequest,
)
from jacobian.math.root_isolation import compare_algebraic, isolate_real_roots


def compute_root_isolation(request: UnivariatePolynomialRequest) -> RootIsolationResult:
    isolated = isolate_real_roots(
        [{"num": c.num, "den": c.den} for c in request.coefficients_descending]
    )
    return RootIsolationResult(
        roots=tuple(
            (
                CanonicalRational(
                    num=format_canonical_integer(sympy.Rational(lower).p),
                    den=format_canonical_integer(sympy.Rational(lower).q),
                ),
                CanonicalRational(
                    num=format_canonical_integer(sympy.Rational(upper).p),
                    den=format_canonical_integer(sympy.Rational(upper).q),
                ),
            )
            for (lower, upper), _multiplicity in isolated
        ),
        multiplicities=tuple(multiplicity for _interval, multiplicity in isolated),
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
