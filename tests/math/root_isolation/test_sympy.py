"""Boundary cases for shared exact SymPy root-isolation primitives."""

import sympy

from jacobian.math._root_isolation import strict_root_count


def _polynomial() -> sympy.Poly:
    x = sympy.Symbol("x")
    return sympy.Poly(x * (x - 1) * (x - 2), x, domain=sympy.ZZ)


def test_strict_root_count_uses_singletons_for_rational_roots() -> None:
    polynomial = _polynomial()

    assert strict_root_count(polynomial, 1, 1) == 1
    assert (
        strict_root_count(polynomial, sympy.Rational(1, 2), sympy.Rational(1, 2)) == 0
    )


def test_strict_root_count_excludes_roots_at_open_interval_endpoints() -> None:
    polynomial = _polynomial()

    assert strict_root_count(polynomial, 0, 2) == 1
    assert strict_root_count(polynomial, -1, 1) == 1
