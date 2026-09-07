"""Typed wire contracts for approximation theory operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import (
    CanonicalRational,
    ExactInteger,
    canonical_rational_component_digits,
    require_bounded_rational,
)
from jacobian._models import StrictModel
from jacobian.math.polynomials.values import RationalPolynomial

# Barycentric weights are products of node differences; their components can
# grow to roughly the sum of per-node component digit budgets plus small
# polynomial factors. Capping that sum keeps every derived basis coefficient
# and barycentric weight inside the canonical 32,768-digit limit.
MAX_NODE_COMPONENT_DIGITS_TOTAL = 512
MAX_INTERPOLATION_VALUE_DIGITS = 256
MAX_INTERPOLATION_NODES = 32


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable validation error owned by approximation contracts."""

    return PydanticCustomError(f"approximation_theory.{reason}", message)


class RationalNodeSet(StrictModel):
    """A finite set of distinct rational interpolation nodes in increasing order."""

    nodes: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_INTERPOLATION_NODES
    )

    @model_validator(mode="after")
    def require_distinct_sorted(self) -> Self:
        fracs = [n.as_fraction() for n in self.nodes]
        if len(fracs) != len(set(fracs)):
            raise _validation_error(
                "nodes_not_distinct", "interpolation nodes must be distinct"
            )
        if fracs != sorted(fracs):
            raise _validation_error(
                "nodes_not_increasing",
                "interpolation nodes must be in increasing order",
            )
        if (
            sum(canonical_rational_component_digits(node) for node in self.nodes)
            > MAX_NODE_COMPONENT_DIGITS_TOTAL
        ):
            raise _validation_error(
                "node_component_budget_exceeded",
                "nodes exceed the "
                f"{MAX_NODE_COMPONENT_DIGITS_TOTAL}-digit component budget; "
                "derived barycentric weights would leave the canonical range",
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
            raise _validation_error(
                "basis_variable_mismatch",
                "Lagrange basis polynomial must use variable 'x'",
            )
        return self


class LagrangeBasisResult(StrictModel):
    """Lagrange basis polynomials with their retained node set."""

    nodes: RationalNodeSet
    node_count: ExactInteger = Field(ge=1, le=MAX_INTERPOLATION_NODES)
    basis: tuple[LagrangeBasisPolynomial, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_bound_to_nodes(self) -> Self:
        if self.node_count != len(self.nodes.nodes):
            raise _validation_error(
                "node_count_mismatch", "node_count must match the retained node set"
            )
        if len(self.basis) != self.node_count:
            raise _validation_error(
                "basis_length_mismatch", "node_count must match basis length"
            )
        indices = sorted(entry.index for entry in self.basis)
        if indices != list(range(self.node_count)):
            raise _validation_error(
                "basis_indices_invalid",
                "basis indices must be exactly 0..node_count-1 with no repeats",
            )
        return self


class LagrangeInterpolationRequest(StrictModel):
    """Interpolate values at nodes using Lagrange interpolation."""

    nodes: RationalNodeSet
    values: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_INTERPOLATION_NODES
    )

    @model_validator(mode="after")
    def require_matching_axes(self) -> Self:
        if len(self.nodes.nodes) != len(self.values):
            raise _validation_error(
                "interpolation_length_mismatch",
                "values must have the same length as nodes",
            )
        return self


class LagrangeInterpolationData(StrictModel):
    """The canonical node/value axes defining a Lagrange interpolant."""

    nodes: RationalNodeSet
    values: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_INTERPOLATION_NODES
    )

    @model_validator(mode="after")
    def require_matching_axes(self) -> Self:
        if len(self.nodes.nodes) != len(self.values):
            raise _validation_error(
                "interpolation_length_mismatch",
                "values must have the same length as nodes",
            )
        return self


def admit_interpolation_values(
    nodes: RationalNodeSet, values: tuple[CanonicalRational, ...]
) -> None:
    """Apply the shared canonical interpolation envelope before its kernel."""

    if len(values) != len(nodes.nodes):
        raise _validation_error(
            "interpolation_length_mismatch",
            "values must have the same length as nodes",
        )
    for value in values:
        try:
            require_bounded_rational(
                value,
                max_digits=MAX_INTERPOLATION_VALUE_DIGITS,
                label="interpolation value",
            )
        except ValueError as exc:
            raise _validation_error("interpolation_value_too_large", str(exc)) from exc


class LagrangeInterpolationResult(StrictModel):
    """The interpolation polynomial as a canonical rational polynomial."""

    source: LagrangeInterpolationData
    polynomial: RationalPolynomial

    @model_validator(mode="after")
    def require_interpolation_variable(self) -> Self:
        if self.polynomial.variables != ("x",):
            raise _validation_error(
                "interpolation_variable_mismatch",
                "interpolation polynomial must use the single variable 'x'",
            )
        return self

    @classmethod
    def _from_kernel(
        cls, *, source: LagrangeInterpolationData, polynomial: RationalPolynomial
    ) -> Self:
        return cls.model_construct(source=source, polynomial=polynomial)

    @property
    def degree(self) -> int:
        if not self.polynomial.polynomial.terms:
            return 0
        return max(term.exponents[0] for term in self.polynomial.polynomial.terms)


__all__ = [
    "LagrangeBasisPolynomial",
    "LagrangeBasisRequest",
    "LagrangeBasisResult",
    "LagrangeInterpolationData",
    "LagrangeInterpolationRequest",
    "LagrangeInterpolationResult",
    "RationalNodeSet",
]
