"""Domain-owned arithmetic dynamics operations."""

from __future__ import annotations

from fractions import Fraction
from functools import reduce
from math import gcd
from typing import Any

import sympy

from jacobian.math.arithmetic_dynamics._models import (
    CycleMultiplierRequest,
    CycleMultiplierResult,
    DynatomicPolynomialRequest,
    DynatomicPolynomialResult,
    FiniteFieldMapRequest,
    FiniteFieldMapResult,
    FixedPointEquationRequest,
    FixedPointEquationResult,
    MapIterateRequest,
    MapIterateResult,
    OrbitPrefixRequest,
    OrbitPrefixResult,
    PolynomialMapRequest,
)

_x = sympy.Symbol("x")


def _coeffs_to_poly(coefficients: tuple[str, ...]) -> sympy.Poly:
    """Convert coefficient tuple (low to high) to a sympy Poly."""
    coeffs = [Fraction(c) for c in coefficients]
    poly = sum(c * _x**i for i, c in enumerate(coeffs))
    return sympy.Poly(poly, _x, domain=sympy.QQ)


def _poly_to_coeffs(poly: sympy.Poly) -> tuple[str, ...]:
    """Convert a sympy Poly to a coefficient tuple (low to high)."""
    if poly.is_zero:
        return ("0",)
    degree = poly.degree()
    coeffs = []
    for i in range(degree + 1):
        coeff = poly.all_coeffs()[degree - i] if i <= degree else 0
        if coeff == 0 and i > degree:
            coeffs.append("0")
        else:
            coeffs.append(str(Fraction(coeff)))
    return tuple(coeffs)


def compute_map_iterate(request: MapIterateRequest) -> MapIterateResult:
    """Compute the n-th iterate of a polynomial map by composition."""
    f = _coeffs_to_poly(request.coefficients)
    x = sympy.Symbol("x")
    result = x
    for _ in range(request.n):
        result = f.as_expr().subs(x, result)
    result_poly = sympy.Poly(result, x, domain=sympy.QQ)
    degree = result_poly.degree() if not result_poly.is_zero else 0
    return MapIterateResult(
        n=request.n,
        coefficients=_poly_to_coeffs(result_poly),
        degree=degree,
    )


def compute_orbit_prefix(request: OrbitPrefixRequest) -> OrbitPrefixResult:
    """Compute the orbit prefix of a point under a polynomial map."""
    f = _coeffs_to_poly(request.coefficients)
    expr = f.as_expr()
    start = Fraction(request.start)
    orbit: list[Fraction] = [start]
    first_repeat_index: int | None = None
    first_repeat_match: int | None = None

    x = sympy.Symbol("x")
    current = start
    for i in range(request.length):
        current = Fraction(sympy.simplify(expr.subs(x, sympy.Rational(current.numerator, current.denominator))))
        orbit.append(current)
        # Check for repeat
        for j, prev in enumerate(orbit[:-1]):
            if prev == current:
                first_repeat_index = i + 1
                first_repeat_match = j
                break
        if first_repeat_index is not None:
            break

    return OrbitPrefixResult(
        orbit=tuple(str(f) for f in orbit),
        length=len(orbit) - 1,
        first_repeat_index=first_repeat_index,
        first_repeat_match=first_repeat_match,
    )


def compute_fixed_point_equation(
    request: FixedPointEquationRequest,
) -> FixedPointEquationResult:
    """Compute f^n(x) - x as a polynomial."""
    f = _coeffs_to_poly(request.coefficients)
    x = sympy.Symbol("x")
    result = x
    for _ in range(request.n):
        result = f.as_expr().subs(x, result)
    result_poly = sympy.Poly(result - x, x, domain=sympy.QQ)
    degree = result_poly.degree() if not result_poly.is_zero else 0
    return FixedPointEquationResult(
        coefficients=_poly_to_coeffs(result_poly),
        degree=degree,
    )


def _mobius_mu(n: int) -> int:
    """Mobius function mu(n)."""
    if n == 1:
        return 1
    if n == 0:
        return 0
    factors = set()
    d = n
    p = 2
    while p * p <= d:
        if d % p == 0:
            if p in factors:
                return 0
            factors.add(p)
            d //= p
            while d % p == 0:
                d //= p
                if d % p == 0:
                    return 0
            p = 2
        else:
            p += 1
    if d > 1:
        if d in factors:
            return 0
        factors.add(d)
    return (-1) ** len(factors)


def _divisors(n: int) -> list[int]:
    """Return all positive divisors of n."""
    divs = []
    for i in range(1, n + 1):
        if n % i == 0:
            divs.append(i)
    return divs


def compute_dynatomic_polynomial(
    request: DynatomicPolynomialRequest,
) -> DynatomicPolynomialResult:
    """Compute the n-th dynatomic polynomial.

    Phi*_n(x) = product_{d|n} (f^d(x) - x)^{mu(n/d)}
    """
    f = _coeffs_to_poly(request.coefficients)
    n = request.n
    x = sympy.Symbol("x")

    # Compute iterates
    iterates: dict[int, sympy.Expr] = {}
    for d in _divisors(n):
        expr = x
        for _ in range(d):
            expr = f.as_expr().subs(x, expr)
        iterates[d] = expr - x

    # Compute product
    result: sympy.Expr = sympy.Integer(1)
    for d in _divisors(n):
        mu = _mobius_mu(n // d)
        if mu > 0:
            result *= iterates[d]
        elif mu < 0:
            result /= iterates[d]

    result_poly = sympy.Poly(result, x, domain=sympy.QQ)
    degree = result_poly.degree() if not result_poly.is_zero else 0
    return DynatomicPolynomialResult(
        coefficients=_poly_to_coeffs(result_poly),
        degree=degree,
        n=n,
    )


def compute_cycle_multiplier(
    request: CycleMultiplierRequest,
) -> CycleMultiplierResult:
    """Compute the multiplier of a periodic cycle.

    The multiplier is the product of f'(P_i) over all cycle points.
    """
    f = _coeffs_to_poly(request.coefficients)
    f_deriv = sympy.Poly(sympy.diff(f.as_expr(), sympy.Symbol("x")), sympy.Symbol("x"), domain=sympy.QQ)
    x = sympy.Symbol("x")

    multiplier = Fraction(1)
    for point_str in request.cycle:
        point = Fraction(point_str)
        deriv_val = f_deriv.as_expr().subs(x, sympy.Rational(point.numerator, point.denominator))
        multiplier *= Fraction(deriv_val)

    return CycleMultiplierResult(
        multiplier=str(multiplier),
        cycle=request.cycle,
    )


def compute_finite_field_map(request: FiniteFieldMapRequest) -> FiniteFieldMapResult:
    """Compute the functional graph of a polynomial map over GF(p)."""
    p = request.prime
    coeffs = [int(c) % p for c in request.coefficients]

    def f(x: int) -> int:
        result = 0
        power = 1
        for coeff in coeffs:
            result = (result + coeff * power) % p
            power = (power * x) % p
        return result

    # Compute functional graph
    edges: list[tuple[int, int]] = []
    for x in range(p):
        edges.append((x, f(x)))

    # Find cycles and tail lengths
    visited: set[int] = set()
    cycles: list[tuple[int, ...]] = []
    tail_lengths: list[int] = [0] * p

    for start in range(p):
        if start in visited:
            continue
        path: list[int] = [start]
        path_pos: dict[int, int] = {start: 0}
        current = start
        while True:
            nxt = f(current)
            if nxt in path_pos:
                # Found a cycle
                cycle_start = path_pos[nxt]
                cycle = tuple(path[cycle_start:])
                if cycle not in cycles and len(cycle) > 0:
                    cycles.append(cycle)
                # Tail lengths
                for i, node in enumerate(path):
                    if i < cycle_start:
                        tail_lengths[node] = cycle_start - i
                    visited.add(node)
                break
            if nxt in visited:
                for i, node in enumerate(path):
                    tail_lengths[node] = len(path) - i + tail_lengths[nxt]
                    visited.add(node)
                break
            path.append(nxt)
            path_pos[nxt] = len(path) - 1
            current = nxt

    return FiniteFieldMapResult(
        prime=p,
        edges=tuple(edges),
        cycles=tuple(cycles),
        tail_lengths=tuple(tail_lengths),
    )


__all__ = [
    "compute_cycle_multiplier",
    "compute_dynatomic_polynomial",
    "compute_finite_field_map",
    "compute_fixed_point_equation",
    "compute_map_iterate",
    "compute_orbit_prefix",
]
