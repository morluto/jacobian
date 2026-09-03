"""Exact native and catalog operations for rational interpolation."""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.analysis.approximation._models import (
    LagrangeBasisPolynomial,
    LagrangeBasisResult,
    RationalNodeSet,
    admit_interpolation_values,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _polynomial_from_coeffs(coeffs: Sequence[Fraction]) -> RationalPolynomial:
    terms = tuple(
        RationalPolynomialTerm(
            coefficient=CanonicalRational.from_fraction(coeff), exponents=(exp,)
        )
        for exp, coeff in sorted(
            ((exp, coeff) for exp, coeff in enumerate(coeffs) if coeff),
            reverse=True,
        )
    )
    return RationalPolynomial(
        variables=("x",), polynomial=SparseRationalPolynomial(terms=terms)
    )


def _poly_multiply(a: Sequence[Fraction], b: Sequence[Fraction]) -> list[Fraction]:
    result = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            result[i + j] += ai * bj
    return result


def _poly_scale(a: Sequence[Fraction], scale: Fraction) -> list[Fraction]:
    return [value * scale for value in a]


def _nodal_polynomial_and_weights(
    nodes: Sequence[Fraction],
) -> tuple[list[Fraction], list[Fraction]]:
    """Return ``prod(x-x_i)`` and the exact barycentric weights."""

    polynomial = [Fraction(1)]
    denominators = [Fraction(1) for _ in nodes]
    for right, right_node in enumerate(nodes):
        polynomial = _poly_multiply(polynomial, [-right_node, Fraction(1)])
        for left in range(right):
            difference = nodes[left] - right_node
            denominators[left] *= difference
            denominators[right] *= -difference
    return polynomial, [Fraction(1) / denominator for denominator in denominators]


def _divide_by_nodal_factor(
    polynomial: Sequence[Fraction], root: Fraction
) -> list[Fraction]:
    """Divide an ascending coefficient vector exactly by ``x - root``."""

    quotient = [Fraction(0)] * (len(polynomial) - 1)
    quotient[-1] = polynomial[-1]
    for degree in range(len(quotient) - 2, -1, -1):
        quotient[degree] = polynomial[degree + 1] + root * quotient[degree + 1]
    assert polynomial[0] + root * quotient[0] == 0
    return quotient


def _interpolate(
    node_set: RationalNodeSet, values: tuple[CanonicalRational, ...]
) -> RationalPolynomial:
    admit_interpolation_values(node_set, values)
    nodes = [node.as_fraction() for node in node_set.nodes]
    samples = [value.as_fraction() for value in values]
    nodal_polynomial, weights = _nodal_polynomial_and_weights(nodes)
    result = [Fraction(0)] * len(nodes)
    for node, sample, weight in zip(nodes, samples, weights, strict=True):
        scale = sample * weight
        quotient = _divide_by_nodal_factor(nodal_polynomial, node)
        for degree, coefficient in enumerate(quotient):
            result[degree] += scale * coefficient
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return _polynomial_from_coeffs(result)


def lagrange_interpolate(
    nodes: Sequence[CanonicalRational], values: Sequence[CanonicalRational]
) -> RationalPolynomial:
    try:
        return _interpolate(RationalNodeSet(nodes=tuple(nodes)), tuple(values))
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=("nodes", "values"), code=exc.type, message=exc.message()
        ) from exc
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("nodes", "values"),
            code="approximation.interpolation_invalid_domain",
            message=str(exc),
        ) from exc


def lagrange_basis(nodes: RationalNodeSet) -> LagrangeBasisResult:
    """Return the exact Lagrange basis and barycentric weights for ``nodes``."""

    rational_nodes = [node.as_fraction() for node in nodes.nodes]
    nodal_polynomial, weights = _nodal_polynomial_and_weights(rational_nodes)
    basis = []
    for k, (node, weight) in enumerate(zip(rational_nodes, weights, strict=True)):
        polynomial = _divide_by_nodal_factor(nodal_polynomial, node)
        basis.append(
            LagrangeBasisPolynomial(
                index=k,
                polynomial=_polynomial_from_coeffs(_poly_scale(polynomial, weight)),
                barycentric_weight=CanonicalRational.from_fraction(weight),
            )
        )
    return LagrangeBasisResult(
        nodes=nodes, node_count=len(rational_nodes), basis=tuple(basis)
    )


__all__ = [
    "lagrange_basis",
    "lagrange_interpolate",
]
