"""Domain-owned approximation theory operations."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.approximation_theory._models import (
    LagrangeBasisPolynomial,
    LagrangeBasisRequest,
    LagrangeBasisResult,
    LagrangeInterpolationRequest,
    LagrangeInterpolationResult,
    RationalNodeSet,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _polynomial_from_coeffs(coeffs: list[Fraction]) -> RationalPolynomial:
    """Convert dense coefficient list [a0, a1, ...] to RationalPolynomial on 'x'."""
    terms = []
    for exp, coeff in enumerate(coeffs):
        if coeff == 0:
            continue
        terms.append(
            RationalPolynomialTerm(
                coefficient=CanonicalRational.from_fraction(coeff),
                exponents=(exp,),
            )
        )
    # Ensure descending lexicographic order (highest exponent first)
    terms.sort(key=lambda t: t.exponents, reverse=True)
    if not terms:
        # Zero polynomial is represented as empty term tuple
        return RationalPolynomial(
            variables=("x",),
            polynomial=SparseRationalPolynomial(terms=()),
        )
    return RationalPolynomial(
        variables=("x",),
        polynomial=SparseRationalPolynomial(terms=tuple(terms)),
    )


def _poly_multiply(a: list[Fraction], b: list[Fraction]) -> list[Fraction]:
    """Multiply two polynomials represented as coefficient lists."""
    result = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            result[i + j] += ai * bj
    return result


def _poly_add(a: list[Fraction], b: list[Fraction]) -> list[Fraction]:
    """Add two polynomials."""
    n = max(len(a), len(b))
    result = [Fraction(0)] * n
    for i in range(len(a)):
        result[i] += a[i]
    for i in range(len(b)):
        result[i] += b[i]
    return result


def _poly_scale(a: list[Fraction], scale: Fraction) -> list[Fraction]:
    """Scale a polynomial by a constant."""
    return [c * scale for c in a]


def compute_lagrange_basis(request: LagrangeBasisRequest) -> LagrangeBasisResult:
    """Compute the Lagrange basis polynomials and barycentric weights.

    For nodes x_0, ..., x_{n-1}, the k-th Lagrange basis polynomial is:
    l_k(x) = product_{i != k} (x - x_i) / (x_k - x_i)

    The barycentric weight is:
    w_k = 1 / product_{i != k} (x_k - x_i)
    """
    nodes = [n.as_fraction() for n in request.nodes.nodes]
    n = len(nodes)

    basis_polys = []
    for k in range(n):
        x_k = nodes[k]

        poly = [Fraction(1)]
        denom = Fraction(1)

        for i in range(n):
            if i == k:
                continue
            x_i = nodes[i]
            linear_factor = [-x_i, Fraction(1)]
            poly = _poly_multiply(poly, linear_factor)
            denom *= x_k - x_i

        bary_weight = Fraction(1) / denom

        # Normalize the polynomial by the denominator to get l_k(x)
        poly = _poly_scale(poly, Fraction(1) / denom)
        basis_polys.append(
            LagrangeBasisPolynomial(
                index=k,
                polynomial=_polynomial_from_coeffs(poly),
                barycentric_weight=CanonicalRational.from_fraction(bary_weight),
            )
        )

    return LagrangeBasisResult(
        nodes=request.nodes,
        node_count=n,
        basis=tuple(basis_polys),
    )


def compute_lagrange_interpolation(
    request: LagrangeInterpolationRequest,
) -> LagrangeInterpolationResult:
    """Interpolate a polynomial through given nodes and values.

    Uses the Lagrange formula: p(x) = sum_k y_k * l_k(x)
    where l_k(x) is the k-th Lagrange basis polynomial.
    """
    return _interpolate(request.nodes, request.values)


def _interpolate(
    node_set: RationalNodeSet, values: tuple[CanonicalRational, ...]
) -> LagrangeInterpolationResult:
    """Compute from admitted canonical values shared by native and wire paths."""

    nodes = [node.as_fraction() for node in node_set.nodes]
    interpolation_values = [value.as_fraction() for value in values]
    n = len(nodes)

    result_poly = [Fraction(0)]

    for k in range(n):
        x_k = nodes[k]
        poly = [Fraction(1)]

        for i in range(n):
            if i == k:
                continue
            x_i = nodes[i]
            linear_factor = [-x_i, Fraction(1)]
            poly = _poly_multiply(poly, linear_factor)

        for i in range(n):
            if i == k:
                continue
            x_i = nodes[i]
            poly = _poly_scale(poly, Fraction(1) / (x_k - x_i))

        scaled_poly = _poly_scale(poly, interpolation_values[k])
        result_poly = _poly_add(result_poly, scaled_poly)

    while len(result_poly) > 1 and result_poly[-1] == 0:
        result_poly.pop()

    return LagrangeInterpolationResult(
        polynomial=_polynomial_from_coeffs(result_poly),
    )


__all__ = [
    "compute_lagrange_basis",
    "compute_lagrange_interpolation",
]
