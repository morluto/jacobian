"""Typed wire contracts for exact bounded graphical-model operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.graphical_models._validation import validate_d_separation_input
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
            raise PydanticCustomError(
                "graphical_model.factor_domains_mismatch",
                "factors must share the exact model domain_sizes",
            )
        return self


class FactorMultiplyResult(FactorMultiplyRequest):
    factor: Factor

    @classmethod
    def _from_kernel(cls, left: Factor, right: Factor, factor: Factor) -> Self:
        """Construct trusted output from the owner-local exact kernel."""

        return cls.model_construct(left=left, right=right, factor=factor)


class FactorMarginalizeRequest(StrictModel):
    factor: Factor
    variable: Variable

    @model_validator(mode="after")
    def require_valid_variable(self) -> Self:
        if self.variable not in self.factor.variables:
            raise PydanticCustomError(
                "graphical_model.factor_variable_missing",
                "variable is not in factor",
            )
        return self


class FactorMarginalizeResult(StrictModel):
    source_factor: Factor
    variable: Variable
    factor: Factor

    @classmethod
    def _from_kernel(
        cls, source_factor: Factor, variable: Variable, factor: Factor
    ) -> Self:
        """Construct trusted output from the owner-local exact kernel."""

        return cls.model_construct(
            source_factor=source_factor, variable=variable, factor=factor
        )


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
        try:
            validate_d_separation_input(
                self.variable_count,
                self.edges,
                self.set_a,
                self.set_b,
                self.set_c,
            )
        except ValueError as error:
            raise PydanticCustomError(
                "graphical_model.d_separation_invalid",
                str(error),
            ) from error
        return self


class DSeparationResult(DSeparationRequest):
    d_separated: bool

    @classmethod
    def _from_kernel(cls, request: DSeparationRequest, d_separated: bool) -> Self:
        """Construct trusted output from the owner-local exact kernel."""

        return cls.model_construct(
            variable_count=request.variable_count,
            edges=request.edges,
            set_a=request.set_a,
            set_b=request.set_b,
            set_c=request.set_c,
            d_separated=d_separated,
        )


__all__ = [
    "DSeparationRequest",
    "DSeparationResult",
    "FactorMarginalizeRequest",
    "FactorMarginalizeResult",
    "FactorMultiplyRequest",
    "FactorMultiplyResult",
]
