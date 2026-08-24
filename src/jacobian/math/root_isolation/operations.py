"""Root isolation backed by SymPy."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from jacobian.canonical import parse_canonical_integer
from jacobian.math.real_algebraic import (
    RealAlgebraicOrderValue,
    RealAlgebraicValue,
    compare_real_algebraic,
)

__all__ = ["compare_algebraic", "isolate_real_roots"]


def _polynomial(coeffs_desc: Sequence[dict[str, str]]) -> Any:
    import sympy

    x = sympy.Symbol("x")
    expression = sum(
        sympy.Rational(
            parse_canonical_integer(c["num"]), parse_canonical_integer(c["den"])
        )
        * sympy.Symbol("x") ** i
        for i, c in enumerate(reversed(coeffs_desc))
    )
    return sympy.Poly(expression, x, domain=sympy.QQ)


def isolate_real_roots(coeffs_desc: Sequence[dict[str, str]]) -> Any:
    """Return SymPy's exact rational isolating intervals and multiplicities."""
    return _polynomial(coeffs_desc).intervals()


def compare_algebraic(
    left: RealAlgebraicValue,
    right: RealAlgebraicValue,
) -> RealAlgebraicOrderValue:
    """Compare canonical real algebraic values exactly."""

    return compare_real_algebraic(left, right)
