"""Typed wire contracts for polynomial interpolation operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_POINTS = 32


class DividedDifferencesRequest(StrictModel):
    """Compute divided differences from sample points."""

    nodes: tuple[str, ...] = Field(min_length=1, max_length=MAX_POINTS)
    values: tuple[str, ...] = Field(min_length=1, max_length=MAX_POINTS)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if len(self.nodes) != len(self.values):
            raise ValueError("nodes and values must have the same length")
        return self


class NewtonFormRequest(StrictModel):
    """Compute Newton form coefficients from divided differences."""

    nodes: tuple[str, ...] = Field(min_length=1, max_length=MAX_POINTS)
    values: tuple[str, ...] = Field(min_length=1, max_length=MAX_POINTS)


class NewtonEvaluateRequest(StrictModel):
    """Evaluate a polynomial in Newton form at a point."""

    nodes: tuple[str, ...] = Field(min_length=1, max_length=MAX_POINTS)
    values: tuple[str, ...] = Field(min_length=1, max_length=MAX_POINTS)
    evaluation_point: str = Field(min_length=1)


# Results


class DividedDifferencesResult(StrictModel):
    coefficients: tuple[str, ...]
    method: str = "NEWTON_DIVIDED_DIFFERENCES"


class NewtonFormResult(StrictModel):
    coefficients: tuple[str, ...]
    nodes: tuple[str, ...]
    method: str = "NEWTON_FORM"


class NewtonEvaluateResult(StrictModel):
    result: str
    method: str = "NEWTON_HORNER"
