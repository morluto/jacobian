"""Typed contracts for bounded constraint-satisfaction object construction."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian.contracts.base import ContractModel
from jacobian.contracts.exact import CanonicalRational


class IntegerLinearConstraint(ContractModel):
    """One linear constraint over integer variables: sum(a_i * x_i) <= b."""

    coefficients: tuple[StrictInt, ...] = Field(min_length=1, max_length=32)
    rhs: StrictInt
    relation: Literal["LE", "EQ", "GE"] = "LE"


class IntegerFeasibilityRequest(ContractModel):
    """Request to find one feasible integer point satisfying all constraints."""

    variable_count: StrictInt = Field(ge=1, le=16)
    constraints: tuple[IntegerLinearConstraint, ...] = Field(
        min_length=1, max_length=64
    )
    timeout_ms: StrictInt = Field(default=5000, ge=100, le=30000)

    @model_validator(mode="after")
    def validate_constraint_dimensions(self) -> Self:
        for i, c in enumerate(self.constraints):
            if len(c.coefficients) != self.variable_count:
                raise ValueError(
                    f"constraint {i} has {len(c.coefficients)} coefficients "
                    f"but variable_count is {self.variable_count}"
                )
        return self


class IntegerFeasibilityResult(ContractModel):
    """Result of a bounded integer feasibility search."""

    status: Literal["FEASIBLE", "INFEASIBLE", "UNKNOWN", "INVALID"]
    assignment: tuple[StrictInt, ...] | None = None
    detail: str = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def bind_status_to_assignment(self) -> Self:
        if self.status == "FEASIBLE" and self.assignment is None:
            raise ValueError("a feasible result requires an assignment")
        if self.status != "FEASIBLE" and self.assignment is not None:
            raise ValueError("only a feasible result may carry an assignment")
        return self


class IntegerFeasibilityCheckRequest(ContractModel):
    """Request to independently verify that an assignment satisfies all constraints."""

    variable_count: StrictInt = Field(ge=1, le=16)
    constraints: tuple[IntegerLinearConstraint, ...] = Field(
        min_length=1, max_length=64
    )
    assignment: tuple[StrictInt, ...] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def validate_dimensions(self) -> Self:
        for i, c in enumerate(self.constraints):
            if len(c.coefficients) != self.variable_count:
                raise ValueError(
                    f"constraint {i} has {len(c.coefficients)} coefficients "
                    f"but variable_count is {self.variable_count}"
                )
        if len(self.assignment) != self.variable_count:
            raise ValueError("assignment length must match variable_count")
        return self


class IntegerFeasibilityCheckResult(ContractModel):
    """Result of independently verifying an integer feasibility assignment."""

    satisfies: bool
    first_violated_constraint: StrictInt | None = None
    detail: str = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def bind_violation(self) -> Self:
        if self.satisfies and self.first_violated_constraint is not None:
            raise ValueError("a satisfying result cannot have a violated constraint")
        return self
