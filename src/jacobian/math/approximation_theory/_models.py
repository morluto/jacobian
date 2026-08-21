"""Typed wire contracts for approximation theory operations."""

from __future__ import annotations

from fractions import Fraction
from typing import Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel

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
    """One Lagrange basis polynomial l_k(x) as coefficients [a_0, a_1, ...]."""

    index: int = Field(ge=0)
    coefficients: tuple[CanonicalRational, ...] = Field(min_length=1)
    barycentric_weight: CanonicalRational


class LagrangeBasisResult(StrictModel):
    """Lagrange basis polynomials and barycentric weights."""

    node_count: int = Field(ge=1, le=32)
    basis: tuple[LagrangeBasisPolynomial, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_consistent_count(self) -> Self:
        if self.node_count != len(self.basis):
            raise ValueError("node_count must match basis length")
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
    """The interpolation polynomial as coefficients [a_0, a_1, ...]."""

    degree: int = Field(ge=0)
    coefficients: tuple[CanonicalRational, ...] = Field(min_length=1)


__all__ = [
    "LagrangeBasisPolynomial",
    "LagrangeBasisRequest",
    "LagrangeBasisResult",
    "LagrangeInterpolationRequest",
    "LagrangeInterpolationResult",
    "RationalNodeSet",
]
