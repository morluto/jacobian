"""Typed wire contracts for exact bounded graphical-model operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.graphical_models.values import (
    MAX_MODEL_VARS,
    Factor,
    Variable,
)


class FactorMultiplyRequest(StrictModel):
    left: Factor
    right: Factor

    @model_validator(mode="after")
    def require_compatible_domains(self) -> Self:
        if self.left.domain_sizes != self.right.domain_sizes:
            raise ValueError("factors must share the exact model domain_sizes")
        return self


class FactorMultiplyResult(FactorMultiplyRequest):
    factor: Factor

    @model_validator(mode="after")
    def bind_product(self) -> Self:
        from jacobian.math.graphical_models.operations import factor_multiply

        if self.factor != factor_multiply(self.left, self.right):
            raise ValueError("factor must be the exact product of the bound operands")
        return self


class FactorMarginalizeRequest(StrictModel):
    factor: Factor
    variable: Variable

    @model_validator(mode="after")
    def require_valid_variable(self) -> Self:
        if self.variable not in self.factor.variables:
            raise ValueError("variable is not in factor")
        return self


class FactorMarginalizeResult(StrictModel):
    source_factor: Factor
    variable: Variable
    factor: Factor

    @model_validator(mode="after")
    def bind_marginal(self) -> Self:
        from jacobian.math.graphical_models.operations import factor_marginalize

        if self.factor != factor_marginalize(self.source_factor, self.variable):
            raise ValueError("factor must be the exact bound marginal")
        return self


class DSeparationRequest(StrictModel):
    variable_count: int = Field(ge=1, le=MAX_MODEL_VARS)
    edges: tuple[tuple[int, int], ...] = Field(
        default=(), max_length=MAX_MODEL_VARS * (MAX_MODEL_VARS - 1) // 2
    )
    set_a: tuple[Variable, ...] = Field(min_length=1, max_length=MAX_MODEL_VARS)
    set_b: tuple[Variable, ...] = Field(min_length=1, max_length=MAX_MODEL_VARS)
    set_c: tuple[Variable, ...] = Field(default=(), max_length=MAX_MODEL_VARS)

    @model_validator(mode="after")
    def require_dag_and_disjoint_sets(self) -> Self:
        from jacobian.math.graphical_models.operations import (
            validate_d_separation_input,
        )

        validate_d_separation_input(
            self.variable_count,
            self.edges,
            self.set_a,
            self.set_b,
            self.set_c,
        )
        return self


class DSeparationResult(DSeparationRequest):
    d_separated: bool

    @model_validator(mode="after")
    def bind_decision(self) -> Self:
        from jacobian.math.graphical_models.operations import d_separation

        expected = d_separation(
            self.variable_count,
            self.edges,
            self.set_a,
            self.set_b,
            self.set_c,
        )
        if self.d_separated != expected:
            raise ValueError("decision must match the bound d-separation instance")
        return self


__all__ = [
    "DSeparationRequest",
    "DSeparationResult",
    "FactorMarginalizeRequest",
    "FactorMarginalizeResult",
    "FactorMultiplyRequest",
    "FactorMultiplyResult",
]
