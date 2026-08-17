"""SymPy boundary checks for authoritative sparse rational polynomials."""

from __future__ import annotations

import pytest
from sympy import Poly, symbols

from jacobian.math.polynomial._conversions import rational_polynomial_from_sympy


def test_sympy_conversion_rejects_non_qq_or_reordered_generators() -> None:
    x, y = symbols("x y")

    with pytest.raises(ValueError, match="exact QQ"):
        rational_polynomial_from_sympy(Poly(x + 1, x, domain="ZZ"), ("x",))
    with pytest.raises(ValueError, match="declared order"):
        rational_polynomial_from_sympy(Poly(x + y, x, y, domain="QQ"), ("y", "x"))
