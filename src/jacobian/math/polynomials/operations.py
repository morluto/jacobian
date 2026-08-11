"""Exact polynomial operations on canonical SymPy ``Poly`` inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sympy import Poly

__all__ = ["derivative", "gcdex", "resultant"]


def _poly(value: Poly) -> Poly:
    from sympy import Poly

    if not isinstance(value, Poly):
        raise TypeError("polynomial must be a SymPy Poly")
    return value


def gcdex(left: Poly, right: Poly) -> tuple[Poly, Poly, Poly]:
    """Return the exact extended-GCD tuple for two compatible polynomials."""

    from jacobian.math.polynomials import _sympy

    return _sympy.polynomial_gcdex(_poly(left), _poly(right))


def resultant(left: Poly, right: Poly, generator: Any) -> Any:
    """Return the exact resultant in the supplied common generator."""

    from jacobian.math.polynomials import _sympy

    return _sympy.polynomial_resultant(_poly(left), _poly(right), generator)


def derivative(polynomial: Poly) -> Poly:
    """Return the formal derivative of a polynomial."""

    from jacobian.math.polynomials import _sympy

    return _sympy.polynomial_derivative(_poly(polynomial))
