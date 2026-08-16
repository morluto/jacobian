"""Root isolation backed by SymPy."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

__all__ = ["compare_algebraic", "isolate_real_roots"]


def _polynomial(coeffs_desc: Sequence[dict[str, str]]) -> Any:
    import sympy

    x = sympy.Symbol("x")
    expression = sum(
        sympy.Rational(c["num"], c["den"]) * sympy.Symbol("x") ** i
        for i, c in enumerate(reversed(coeffs_desc))
    )
    return sympy.Poly(expression, x, domain=sympy.QQ)


def isolate_real_roots(coeffs_desc: Sequence[dict[str, str]]) -> Any:
    """Return SymPy's exact rational isolating intervals and multiplicities."""
    return _polynomial(coeffs_desc).intervals()


def compare_algebraic(
    left_poly: Sequence[dict[str, str]],
    left_lower: Any,
    left_upper: Any,
    right_poly: Sequence[dict[str, str]],
    right_lower: Any,
    right_upper: Any,
) -> Literal["LT", "EQ", "GT"]:
    import sympy

    def selected_real_root(poly: Any, lower: Any, upper: Any) -> Any:
        roots = {
            root for root in poly.all_roots() if root.is_real and lower <= root <= upper
        }
        if len(roots) != 1:
            raise ValueError("isolating interval must contain exactly one real root")
        return roots.pop()

    left_lower = sympy.Rational(left_lower.num, left_lower.den)
    left_upper = sympy.Rational(left_upper.num, left_upper.den)
    right_lower = sympy.Rational(right_lower.num, right_lower.den)
    right_upper = sympy.Rational(right_upper.num, right_upper.den)
    lr = selected_real_root(_polynomial(left_poly), left_lower, left_upper)
    rr = selected_real_root(_polynomial(right_poly), right_lower, right_upper)
    cmp = sympy.sign(lr - rr)
    if cmp < 0:
        return "LT"
    if cmp > 0:
        return "GT"
    return "EQ"
