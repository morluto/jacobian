"""Exact Sturm chain and root counting kernels backed by SymPy."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

__all__ = ["root_count", "sturm_chain"]


def _to_sympy_poly(terms: list[tuple[Fraction, int]]) -> Any:
    from sympy import Poly, Rational, Symbol

    x = Symbol("x")
    poly_dict = {exp: Rational(val.numerator, val.denominator) for val, exp in terms}
    return Poly(poly_dict, x, domain="QQ")


def sturm_chain(terms: list[tuple[Fraction, int]]) -> list[list[tuple[Fraction, int]]]:
    """Compute the Sturm chain of a univariate polynomial."""
    from sympy import sturm

    poly = _to_sympy_poly(terms)
    chain = sturm(poly)
    result = []
    for p in chain:
        result.append(_sympy_poly_to_terms(p))
    return result


def root_count(
    terms: list[tuple[Fraction, int]],
    lower: Fraction,
    upper: Fraction,
) -> int:
    """Count real roots in [lower, upper] using the Sturm theorem."""
    from sympy import Rational

    poly = _to_sympy_poly(terms)
    chain = _build_sturm_chain(poly)

    a = Rational(lower.numerator, lower.denominator)
    b = Rational(upper.numerator, upper.denominator)

    sign_changes_a = _sign_changes(chain, a)
    sign_changes_b = _sign_changes(chain, b)

    return sign_changes_a - sign_changes_b


def _build_sturm_chain(poly: Any) -> list[Any]:
    from sympy import sturm

    return list(sturm(poly))


def _sign_changes(chain: list[Any], point: Any) -> int:

    if len(chain) == 0:
        return 0
    signs = []
    for poly in chain:
        val = poly.as_expr().subs(poly.gen, point)
        if val != 0:
            signs.append(1 if val > 0 else -1)
    count = 0
    for i in range(1, len(signs)):
        if signs[i] != signs[i - 1]:
            count += 1
    return count


def _sympy_poly_to_terms(poly: Any) -> list[tuple[Fraction, int]]:
    from fractions import Fraction

    result = []
    for exps, coeff in poly.as_dict().items():
        if coeff == 0:
            continue
        if hasattr(coeff, "p") and hasattr(coeff, "q"):
            frac = Fraction(int(coeff.p), int(coeff.q))
        else:
            frac = Fraction(coeff)
        result.append((frac, exps[0]))
    return result
