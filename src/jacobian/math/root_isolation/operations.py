"""Root isolation backed by SymPy."""

from __future__ import annotations

__all__ = ["compare_algebraic", "isolate_real_roots"]


def isolate_real_roots(coeffs_desc):
    import sympy

    poly = sum(
        sympy.Rational(c["num"], c["den"]) * sympy.Symbol("x") ** i
        for i, c in enumerate(reversed(coeffs_desc))
    )
    roots = sympy.Poly(poly, sympy.Symbol("x")).all_roots()
    real_roots = [(r, 1) for r in roots if r.is_real]
    return real_roots


def compare_algebraic(
    left_poly, left_lower, left_upper, right_poly, right_lower, right_upper
):
    import sympy

    x = sympy.Symbol("x")
    lp = sum(
        sympy.Rational(c["num"], c["den"]) * x**i
        for i, c in enumerate(reversed(left_poly))
    )
    rp = sum(
        sympy.Rational(c["num"], c["den"]) * x**i
        for i, c in enumerate(reversed(right_poly))
    )
    lr = sympy.CRootOf(sympy.Poly(lp, x), 0)
    for _i in range(len(sympy.Poly(lp, x).all_roots())):
        for r in sympy.Poly(lp, x).all_roots():
            if r.is_real and sympy.nsimplify(left_lower) <= r <= sympy.nsimplify(
                left_upper
            ):
                lr = r
                break
    rr = sympy.CRootOf(sympy.Poly(rp, x), 0)
    for _i in range(len(sympy.Poly(rp, x).all_roots())):
        for r in sympy.Poly(rp, x).all_roots():
            if r.is_real and sympy.nsimplify(right_lower) <= r <= sympy.nsimplify(
                right_upper
            ):
                rr = r
                break
    cmp = sympy.sign(lr - rr)
    if cmp < 0:
        return "LT"
    elif cmp > 0:
        return "GT"
    return "EQ"
