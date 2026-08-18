"""Typed wire contracts for graphical model operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.graphical_models.values import (
    Factor,
)


class FactorMultiplyRequest(StrictModel):
    """Multiply two factors."""

    left: Factor
    right: Factor

    @model_validator(mode="after")
    def require_compatible_domains(self) -> Self:
        if self.left.domain_sizes != self.right.domain_sizes:
            raise ValueError("factors must share the same domain_sizes")
        return self


class FactorMultiplyResult(StrictModel):
    """The product factor."""

    factor: Factor


class FactorMarginalizeRequest(StrictModel):
    """Marginalize out a variable from a factor."""

    factor: Factor
    variable: int = Field(ge=0)

    @model_validator(mode="after")
    def require_valid_variable(self) -> Self:
        if self.variable not in self.factor.variables:
            raise ValueError("variable is not in factor")
        return self


class FactorMarginalizeResult(StrictModel):
    """The marginalized factor."""

    factor: Factor


class VariableEliminationRequest(StrictModel):
    """Compute a marginal via variable elimination.

    ``factors`` are the factor family, ``domain_sizes`` defines variable
    cardinalities, ``elimination_order`` is the order to eliminate variables,
    and ``query_variables`` are the variables to keep.
    """

    factors: tuple[Factor, ...] = Field(min_length=1)
    domain_sizes: tuple[int, ...] = Field(min_length=1)
    elimination_order: tuple[int, ...] = Field(default=())
    query_variables: tuple[int, ...] = Field(default=())

    @model_validator(mode="after")
    def require_valid_variables(self) -> Self:
        max_var = 0
        for f in self.factors:
            for v in f.variables:
                max_var = max(max_var, v)
        if max(self.query_variables, default=0) >= len(self.domain_sizes):
            raise ValueError("query variable out of range")
        return self


class VariableEliminationResult(StrictModel):
    """The resulting marginal factor."""

    factor: Factor


class DSeparationRequest(StrictModel):
    """Check d-separation: are variables A d-separated from B given C?

    The network is specified by a list of (parent, child) edges over
    variable indices.
    """

    variable_count: int = Field(ge=1)
    edges: tuple[tuple[int, int], ...] = Field(default=())
    set_a: tuple[int, ...] = Field(min_length=1)
    set_b: tuple[int, ...] = Field(min_length=1)
    set_c: tuple[int, ...] = Field(default=())


class DSeparationResult(StrictModel):
    """Whether set_a and set_b are d-separated given set_c."""

    d_separated: bool


__all__ = [
    "DSeparationRequest",
    "DSeparationResult",
    "FactorMarginalizeRequest",
    "FactorMarginalizeResult",
    "FactorMultiplyRequest",
    "FactorMultiplyResult",
    "VariableEliminationRequest",
    "VariableEliminationResult",
]
