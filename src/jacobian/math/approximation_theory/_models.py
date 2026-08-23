"""Typed wire contracts for approximation theory operations."""

from __future__ import annotations

from fractions import Fraction
from typing import Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel
from jacobian.math.polynomials.values import RationalPolynomial

# Barycentric weights are products of node differences; their components can
# grow to roughly the sum of per-node component digit budgets plus small
# polynomial factors. Capping that sum keeps every derived basis coefficient
# and barycentric weight inside the canonical 32,768-digit limit.
MAX_NODE_COMPONENT_DIGITS_TOTAL = 512
MAX_INTERPOLATION_VALUE_DIGITS = 256


def _component_digits(value: CanonicalRational) -> int:
    return max(len(value.num.lstrip("-")), len(value.den))


class RationalNodeSet(StrictModel):
    """A finite set of distinct rational interpolation nodes in increasing order."""

    nodes: tuple[CanonicalRational, ...] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def require_distinct_sorted(self) -> Self:
        fracs = [n.as_fraction() for n in self.nodes]
        if len(fracs) != len(set(fracs)):
            raise ValueError("interpolation nodes must be distinct")
        if fracs != sorted(fracs):
            raise ValueError("interpolation nodes must be in increasing order")
        if (
            sum(_component_digits(node) for node in self.nodes)
            > MAX_NODE_COMPONENT_DIGITS_TOTAL
        ):
            raise ValueError(
                "nodes exceed the "
                f"{MAX_NODE_COMPONENT_DIGITS_TOTAL}-digit component budget; "
                "derived barycentric weights would leave the canonical range"
            )
        return self


class LagrangeBasisRequest(StrictModel):
    """Compute the Lagrange basis polynomials for a node set."""

    nodes: RationalNodeSet


class LagrangeBasisPolynomial(StrictModel):
    """One Lagrange basis polynomial l_k(x) as a canonical rational polynomial."""

    index: int = Field(ge=0)
    polynomial: RationalPolynomial
    barycentric_weight: CanonicalRational

    @model_validator(mode="after")
    def require_polynomial_variable(self) -> Self:
        if self.polynomial.variables != ("x",):
            raise ValueError("Lagrange basis polynomial must use variable 'x'")
        return self

    @property
    def coefficients(self) -> tuple[CanonicalRational, ...]:
        from fractions import Fraction

        terms = {
            term.exponents[0]: term.coefficient.as_fraction()
            for term in self.polynomial.polynomial.terms
        }
        max_exp = max(terms.keys()) if terms else 0
        coeffs = []
        for exp in range(max_exp + 1):
            frac = terms.get(exp, Fraction(0))
            coeffs.append(CanonicalRational.from_fraction(frac))
        return tuple(coeffs)


def _evaluate_single_variable(
    polynomial: RationalPolynomial,
    point: Fraction,
) -> Fraction:
    """Exact evaluation of one canonical single-variable polynomial."""
    value = Fraction(0)
    for term in polynomial.polynomial.terms:
        value += term.coefficient.as_fraction() * point ** term.exponents[0]
    return value


def _require_cardinal_entry(
    entry: LagrangeBasisPolynomial,
    nodes: list[Fraction],
    k: int,
    node_count: int,
) -> None:
    """Pin one cardinal polynomial to its node, weight, and delta property."""
    # Node evaluations alone do not identify the basis: a forged
    # higher-degree polynomial can agree with l_k on every node. The
    # genuine cardinal polynomial has degree below node_count, which
    # together with the delta property pins it uniquely.
    terms = entry.polynomial.polynomial.terms
    if any(term.exponents[0] >= node_count for term in terms):
        raise ValueError("basis polynomial degree must be below node_count")
    expected_weight = Fraction(1)
    for i, x_i in enumerate(nodes):
        if i != k:
            expected_weight /= nodes[k] - x_i
    if entry.barycentric_weight.as_fraction() != expected_weight:
        raise ValueError(
            "barycentric weight must equal "
            "1/prod_{i!=k}(x_k - x_i) on the retained nodes"
        )
    for j, x_j in enumerate(nodes):
        evaluated = _evaluate_single_variable(entry.polynomial, x_j)
        expected_value = Fraction(1) if j == k else Fraction(0)
        if evaluated != expected_value:
            raise ValueError(
                "basis polynomial must satisfy l_k(x_j) = delta_kj "
                "on the retained nodes"
            )


class LagrangeBasisResult(StrictModel):
    """Lagrange basis polynomials bound to their retained node set.

    Indices cover 0..node_count-1 exactly once, every barycentric weight
    equals the exact product 1/prod_{i!=k}(x_k - x_i), and each basis
    polynomial satisfies l_k(x_j) = delta_kj on the retained nodes, so a
    revalidated result can only describe this node set's genuine basis.
    """

    nodes: RationalNodeSet
    node_count: int = Field(ge=1, le=32)
    basis: tuple[LagrangeBasisPolynomial, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_bound_to_nodes(self) -> Self:

        nodes = [n.as_fraction() for n in self.nodes.nodes]
        if self.node_count != len(nodes):
            raise ValueError("node_count must match the retained node set")
        if len(self.basis) != self.node_count:
            raise ValueError("node_count must match basis length")
        indices = sorted(entry.index for entry in self.basis)
        if indices != list(range(self.node_count)):
            raise ValueError(
                "basis indices must be exactly 0..node_count-1 with no repeats"
            )
        for entry in self.basis:
            _require_cardinal_entry(entry, nodes, entry.index, self.node_count)
        return self


class LagrangeInterpolationRequest(StrictModel):
    """Interpolate values at nodes using Lagrange interpolation."""

    nodes: RationalNodeSet
    values: tuple[CanonicalRational, ...] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def require_matching_lengths(self) -> Self:
        if len(self.values) != len(self.nodes.nodes):
            raise ValueError("values must have the same length as nodes")
        for value in self.values:
            require_bounded_rational(
                value,
                max_digits=MAX_INTERPOLATION_VALUE_DIGITS,
                label="interpolation value",
            )
        return self


class LagrangeInterpolationResult(StrictModel):
    """The interpolation polynomial as a canonical rational polynomial."""

    polynomial: RationalPolynomial

    @model_validator(mode="after")
    def require_interpolation_variable(self) -> Self:
        if self.polynomial.variables != ("x",):
            raise ValueError(
                "interpolation polynomial must use the single variable 'x'"
            )
        return self

    @property
    def degree(self) -> int:
        if not self.polynomial.polynomial.terms:
            return 0
        return max(term.exponents[0] for term in self.polynomial.polynomial.terms)

    @property
    def coefficients(self) -> tuple[CanonicalRational, ...]:
        # Provide backwards-compatible dense coefficients for tests, derived from polynomial
        from fractions import Fraction

        terms = {
            term.exponents[0]: term.coefficient.as_fraction()
            for term in self.polynomial.polynomial.terms
        }
        max_exp = max(terms.keys()) if terms else 0
        coeffs = []
        for exp in range(max_exp + 1):
            frac = terms.get(exp, Fraction(0))
            coeffs.append(CanonicalRational.from_fraction(frac))
        return tuple(coeffs)


__all__ = [
    "LagrangeBasisPolynomial",
    "LagrangeBasisRequest",
    "LagrangeBasisResult",
    "LagrangeInterpolationRequest",
    "LagrangeInterpolationResult",
    "RationalNodeSet",
]
