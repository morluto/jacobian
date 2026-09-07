"""Boundary cases for shared exact SymPy root-isolation primitives."""

import sympy

from jacobian._exact import CanonicalRational
from jacobian.math._root_isolation import strict_root_count
from jacobian.math.number_theory.algebraic_numbers.root_isolation import (
    isolate_real_roots,
)


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


def test_isolate_real_roots_accepts_canonical_rational_coefficients() -> None:
    coefficients = [
        CanonicalRational(num=1, den=1),
        CanonicalRational(num=0, den=1),
        CanonicalRational(num=-1, den=1),
    ]

    assert isolate_real_roots(coefficients) == [((-1, -1), 1), ((1, 1), 1)]
