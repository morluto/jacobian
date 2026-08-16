"""Multivariate polynomial operations backed by SymPy."""

from __future__ import annotations

__all__ = ["multivariate_gcd", "multivariate_resultant"]


def _parse(expr_str, variables):  # type: ignore[no-untyped-def]
    import sympy

    syms = [sympy.Symbol(v) for v in variables]
    return sympy.sympify(expr_str, locals=dict(zip(variables, syms, strict=True)))


def multivariate_gcd(left_expr, left_vars, right_expr, right_vars):  # type: ignore[no-untyped-def]
    import sympy

    all_vars = list(dict.fromkeys(list(left_vars) + list(right_vars)))
    syms = [sympy.Symbol(v) for v in all_vars]
    local = dict(zip(all_vars, syms, strict=True))
    left = sympy.sympify(left_expr, locals=local)
    right = sympy.sympify(right_expr, locals=local)
    return str(sympy.gcd(left, right))


def multivariate_resultant(left_expr, left_vars, right_expr, right_vars, eliminate_var):  # type: ignore[no-untyped-def]
    import sympy

    all_vars = list(dict.fromkeys(list(left_vars) + list(right_vars)))
    syms = [sympy.Symbol(v) for v in all_vars]
    local = dict(zip(all_vars, syms, strict=True))
    left = sympy.sympify(left_expr, locals=local)
    right = sympy.sympify(right_expr, locals=local)
    x = sympy.Symbol(eliminate_var)
    return str(sympy.resultant(left, right, x))
