"""Domain-owned root isolation operations."""

from __future__ import annotations

import sympy

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.math.root_isolation import compare_algebraic, isolate_real_roots
from jacobian.math.root_isolation._models import (
    AlgebraicCompareRequest,
    AlgebraicCompareResult,
    RootIsolationResult,
    UnivariatePolynomialRequest,
)


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
    return compare_algebraic(request.left, request.right)
