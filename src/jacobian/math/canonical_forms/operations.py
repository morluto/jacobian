"""Exact canonical-form kernels backed by SymPy polynomial algebra."""

from __future__ import annotations

from itertools import combinations
from typing import Any

__all__ = [
    "characteristic_polynomial",
    "invariant_factors",
    "minimal_polynomial",
    "primary_decomposition",
]


def _sympy_matrix(entries: list[list[Any]]) -> Any:
    from sympy import Matrix

    return Matrix(entries)


def characteristic_polynomial(entries: list[list[Any]]) -> list[Any]:
    """Return the monic characteristic polynomial coefficients [a_0, ..., a_n]."""
    from sympy import Symbol

    x = Symbol("x")
    matrix = _sympy_matrix(entries)
    charpoly = matrix.charpoly(x)
    poly = charpoly.as_expr()
    return list(_poly_to_coeffs(poly, x))


def minimal_polynomial(entries: list[list[Any]]) -> list[Any]:
    """Compute the minimal polynomial via the Krylov/nullspace method.

    Returns the monic minimal polynomial as coefficient list [a_0, ..., a_n].
    """
    from sympy import Matrix, Symbol, eye

    x = Symbol("x")
    n = len(entries)
    if n == 0:
        return [1]

    matrix = _sympy_matrix(entries)

    # Build I, A, A^2, ..., A^n as column vectors in R^(n^2)
    powers = [eye(n)]
    for _ in range(n):
        powers.append(powers[-1] * matrix)

    # Stack as rows of a (n+1) x n^2 matrix
    rows = []
    for mat in powers:
        rows.append([mat[i, j] for i in range(n) for j in range(n)])

    stacked = Matrix(rows).T  # n^2 x (n+1)

    _rref, pivots = stacked.rref()

    # Find the first dependent column (non-pivot)
    degree = n + 1
    for i in range(n + 1):
        if i not in pivots:
            degree = i
            break

    if degree == 0:
        return [1]

    # The null space of the first 'degree' columns gives the coefficients
    submatrix = stacked[:, : degree + 1]
    null_vectors = submatrix.nullspace()

    if not null_vectors:
        return [0] * degree + [1]

    coeffs = null_vectors[0]
    minpoly = sum(coeffs[i] * x**i for i in range(degree + 1))

    # Ensure monic
    from sympy import Poly

    poly = Poly(minpoly, x)
    lc = poly.LC()
    minpoly_monic = minpoly / lc

    return list(_poly_to_coeffs(minpoly_monic, x))


def invariant_factors(entries: list[list[Any]]) -> list[list[Any]]:
    """Compute the non-unit invariant factors over QQ[x].

    Returns a list of monic polynomial coefficient lists, ordered by divisibility:
    f_1 | f_2 | ... | f_s.
    """
    from sympy import Matrix, Poly, Symbol, cancel, eye, gcd

    x = Symbol("x")
    n = len(entries)
    matrix = _sympy_matrix(entries)
    mat = x * eye(n) - matrix

    # Compute determinantal divisors d_0, d_1, ..., d_n
    divisors: list[Any] = [None] * (n + 1)
    divisors[0] = Poly(1, x)

    for k in range(1, n + 1):
        minors_gcd = None
        for rows_idx in combinations(range(n), k):
            for cols_idx in combinations(range(n), k):
                sub = Matrix([[mat[r, c] for c in cols_idx] for r in rows_idx])
                det = sub.det()
                if det == 0:
                    continue
                minors_gcd = det if minors_gcd is None else gcd(minors_gcd, det)
        if minors_gcd is not None:
            poly = Poly(minors_gcd, x)
            lc = poly.LC()
            divisors[k] = poly / lc
        else:
            divisors[k] = Poly(1, x)

    # Compute invariant factors: f_k = d_k / d_{k-1}
    factors = []
    for k in range(n, 0, -1):
        prev = divisors[k - 1]
        curr = divisors[k]
        quotient, remainder = divmod(curr, prev)
        if remainder != 0:
            quotient = Poly(cancel(curr.as_expr() / prev.as_expr()), x)
        if quotient.degree() >= 1:
            factors.append(list(_poly_to_coeffs(quotient.as_expr(), x)))

    factors.reverse()
    return factors


def primary_decomposition(entries: list[list[Any]]) -> list[list[Any]]:
    """Decompose the minimal polynomial into irreducible-power components.

    Returns a list of monic polynomial coefficient lists, one for each
    irreducible factor raised to its multiplicity in the minimal polynomial.
    """
    from sympy import Symbol, factor_list

    x = Symbol("x")
    mp_coeffs = minimal_polynomial(entries)
    mp_expr = _coeffs_to_expr(mp_coeffs, x)

    factored = factor_list(mp_expr, x)
    coeff = factored[0]
    components = []
    for poly, power in factored[1]:
        poly_monic = poly / coeff
        poly_powered = poly_monic**power
        components.append(list(_poly_to_coeffs(poly_powered, x)))

    return components


def _poly_to_coeffs(expr: Any, x: Any) -> list[Any]:
    """Extract monic polynomial coefficients [a_0, a_1, ..., a_n] from a SymPy expression."""
    from sympy import Poly

    poly = Poly(expr, x)
    degree = poly.degree()
    coeffs = []
    for i in range(degree + 1):
        c = poly.as_expr().coeff(x, i)
        coeffs.append(c)
    # Normalize to monic
    if poly.LC() != 1:
        lc = poly.LC()
        coeffs = [c / lc for c in coeffs]
    return coeffs


def _coeffs_to_expr(coeffs: list[Any], x: Any) -> Any:
    """Build a SymPy expression from a coefficient list [a_0, ..., a_n]."""
    return sum(coeffs[i] * x**i for i in range(len(coeffs)))
