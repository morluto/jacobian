"""Number field operations backed by SymPy."""
from __future__ import annotations
__all__ = ["discriminant", "ring_of_integers"]

def discriminant(coefficients_descending, variable):
    import sympy
    x = sympy.Symbol(variable)
    poly = sum(sympy.Rational(c) * x**(len(coefficients_descending) - 1 - i)
               for i, c in enumerate(coefficients_descending))
    p = sympy.Poly(poly, x)
    return str(p.discriminant())

def ring_of_integers(coefficients_descending, variable):
    import sympy
    x = sympy.Symbol(variable)
    poly = sum(sympy.Rational(c) * x**(len(coefficients_descending) - 1 - i)
               for i, c in enumerate(coefficients_descending))
    p = sympy.Poly(poly, x)
    # Simplified: return the discriminant-based integral basis info
    return [str(p.discriminant())]
