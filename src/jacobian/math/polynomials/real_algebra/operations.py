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
    """Compute the exact Sturm subresultant chain of a univariate polynomial."""

    from sympy import sturm

    poly = _to_sympy_poly(terms)
    chain = sturm(poly)
    return [_sympy_poly_to_terms(p) for p in chain]


def root_count(
    terms: list[tuple[Fraction, int]],
    lower: Fraction,
    upper: Fraction,
) -> int:
    """Count distinct real roots in the closed interval [lower, upper]."""

    from sympy import Rational

    poly = _to_sympy_poly(terms)
    chain = _build_sturm_chain(poly)

    a = Rational(lower.numerator, lower.denominator)
    b = Rational(upper.numerator, upper.denominator)

    sign_changes_a = _sign_changes(chain, a)
    sign_changes_b = _sign_changes(chain, b)
    root_at_lower = _evaluates_to_zero(poly, a)

    # The zero-skipping variation is right-continuous, so the difference counts
    # the half-open interval (lower, upper]. Add the root exactly at ``lower``
    # back to obtain the advertised closed interval.
    return sign_changes_a - sign_changes_b + (1 if root_at_lower else 0)


def _build_sturm_chain(poly: Any) -> list[Any]:
    from sympy import sturm

    return list(sturm(poly))


def _sign_changes(chain: list[Any], point: Any) -> int:
    if len(chain) == 0:
        return 0
    signs: list[int] = []
    for poly in chain:
        value = poly.as_expr().subs(poly.gen, point)
        if value != 0:
            signs.append(1 if value > 0 else -1)
    count = 0
    for index in range(1, len(signs)):
        if signs[index] != signs[index - 1]:
            count += 1
    return count


def _evaluates_to_zero(poly: Any, point: Any) -> bool:
    return bool(poly.as_expr().subs(poly.gen, point) == 0)


def _sympy_poly_to_terms(poly: Any) -> list[tuple[Fraction, int]]:
    result: list[tuple[Fraction, int]] = []
    for exps, coeff in poly.as_dict().items():
        if coeff == 0:
            continue
        if hasattr(coeff, "p") and hasattr(coeff, "q"):
            fraction = Fraction(int(coeff.p), int(coeff.q))
        else:
            fraction = Fraction(coeff)
        result.append((fraction, int(exps[0])))
    return result
