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


def _poly_add(a: Sequence[Fraction], b: Sequence[Fraction]) -> list[Fraction]:
    result = [Fraction(0)] * max(len(a), len(b))
    for i, value in enumerate(a):
        result[i] += value
    for i, value in enumerate(b):
        result[i] += value
    return result


def _poly_scale(a: Sequence[Fraction], scale: Fraction) -> list[Fraction]:
    return [value * scale for value in a]


def _interpolate(
    node_set: RationalNodeSet, values: tuple[CanonicalRational, ...]
) -> RationalPolynomial:
    admit_interpolation_values(node_set, values)
    nodes = [node.as_fraction() for node in node_set.nodes]
    samples = [value.as_fraction() for value in values]
    result = [Fraction(0)]
    for k, x_k in enumerate(nodes):
        basis = [Fraction(1)]
        for i, x_i in enumerate(nodes):
            if i != k:
                basis = _poly_multiply(basis, [-x_i, Fraction(1)])
                basis = _poly_scale(basis, Fraction(1) / (x_k - x_i))
        result = _poly_add(result, _poly_scale(basis, samples[k]))
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
    basis = []
    for k, x_k in enumerate(rational_nodes):
        polynomial = [Fraction(1)]
        denominator = Fraction(1)
        for i, x_i in enumerate(rational_nodes):
            if i != k:
                polynomial = _poly_multiply(polynomial, [-x_i, Fraction(1)])
                denominator *= x_k - x_i
        weight = Fraction(1) / denominator
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
