"""Typed wire contracts for polynomial vector calculus operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_VARS = 8
MAX_POLYS = 8


class ScalarFieldRequest(StrictModel):
    """A multivariate polynomial scalar field."""

    variables: tuple[str, ...] = Field(min_length=1, max_length=MAX_VARS)
    polynomial: str = Field(min_length=1, max_length=4096)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if any(not v.isidentifier() for v in self.variables):
            raise ValueError("variable names must be valid identifiers")
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("variable names must be distinct")
        return self


class VectorFieldRequest(StrictModel):
    """A multivariate polynomial vector field."""

    variables: tuple[str, ...] = Field(min_length=1, max_length=MAX_VARS)
    components: tuple[str, ...] = Field(min_length=1, max_length=MAX_POLYS)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if any(not v.isidentifier() for v in self.variables):
            raise ValueError("variable names must be valid identifiers")
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("variable names must be distinct")
        if len(self.components) != len(self.variables):
            raise ValueError(
                "vector field must have one component per variable "
                f"(got {len(self.components)} components, "
                f"{len(self.variables)} variables)"
            )
        return self


class DirectionalDerivativeRequest(StrictModel):
    """Directional derivative of a scalar field along a direction vector."""

    variables: tuple[str, ...] = Field(min_length=1, max_length=MAX_VARS)
    polynomial: str = Field(min_length=1, max_length=4096)
    direction: tuple[str, ...] = Field(min_length=1, max_length=MAX_POLYS)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if any(not v.isidentifier() for v in self.variables):
            raise ValueError("variable names must be valid identifiers")
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("variable names must be distinct")
        if len(self.direction) != len(self.variables):
            raise ValueError("direction vector length must match variables length")
        return self


# Results


class ScalarResult(StrictModel):
    """A scalar polynomial result."""

    result: str
    variables: tuple[str, ...] = ()
    method: str = "SYMPY_GRADIENT"


class VectorResult(StrictModel):
    """A vector polynomial result."""

    components: tuple[str, ...]
    variables: tuple[str, ...] = ()
    method: str = "SYMPY_GRADIENT"
