from __future__ import annotations

from sympy import Poly, symbols

from jacobian.math.polynomials import derivative, gcdex, resultant


def test_native_polynomial_api_uses_exact_sympy_values() -> None:
    x = symbols("x")
    left = Poly(x**2 - 1, x, domain="QQ")
    right = Poly(x - 1, x, domain="QQ")

    left_multiplier, right_multiplier, gcd = gcdex(left, right)
    assert left * left_multiplier + right * right_multiplier == gcd
    assert gcd == right
    assert derivative(left) == Poly(2 * x, x, domain="QQ")
    assert resultant(left, right, x) == 0
