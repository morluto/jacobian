"""Provider-independent values for exact finite graphical models."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_FACTOR_VARS = 16
MAX_VAR_DOMAIN = 32
MAX_FACTOR_TABLE_SIZE = 4096
MAX_BN_VARS = 32


class Factor(StrictModel):
    """A factor (potential) over a set of variables.

    ``variables`` lists the variable indices this factor depends on.
    ``domain_sizes`` gives the cardinality of each variable in the model.
    ``table`` is a flat array of rational strings, indexed by the lexicographic
    order over variable assignments.
    """

    variables: tuple[int, ...] = Field(min_length=0)
    domain_sizes: tuple[int, ...] = Field(min_length=1)
    table: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_valid_factor(self) -> Self:
        if not self.variables:
            raise ValueError("factor must have at least one variable")
        if len(self.variables) > MAX_FACTOR_VARS:
            raise ValueError("too many factor variables")
        for v in self.variables:
            if v < 0:
                raise ValueError("variable index must be non-negative")
        expected_size = 1
        for v in self.variables:
            if v >= len(self.domain_sizes):
                raise ValueError("variable index exceeds domain_sizes")
            expected_size *= self.domain_sizes[v]
        if len(self.table) != expected_size:
            raise ValueError(
                f"table size {len(self.table)} does not match expected {expected_size}"
            )
        return self


class BayesianNetwork(StrictModel):
    """A finite Bayesian network with categorical variables.

    ``parents`` gives the parent set for each variable (as a list of lists).
    ``factors`` gives the conditional probability table (CPT) for each variable.
    ``domain_sizes`` gives the cardinality of each variable.
    """

    variable_count: int = Field(ge=1, le=MAX_BN_VARS)
    domain_sizes: tuple[int, ...] = Field(min_length=1)
    parents: tuple[tuple[int, ...], ...]
    factors: tuple[Factor, ...]

    @model_validator(mode="after")
    def require_valid_bn(self) -> Self:
        if len(self.parents) != self.variable_count:
            raise ValueError("parents must have variable_count entries")
        if len(self.factors) != self.variable_count:
            raise ValueError("factors must have variable_count entries")
        if any(d < 1 for d in self.domain_sizes):
            raise ValueError("domain sizes must be at least 1")
        return self


__all__ = [
    "MAX_BN_VARS",
    "MAX_FACTOR_TABLE_SIZE",
    "MAX_FACTOR_VARS",
    "MAX_VAR_DOMAIN",
    "BayesianNetwork",
    "Factor",
]
