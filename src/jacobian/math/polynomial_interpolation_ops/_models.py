"""Typed wire contracts for polynomial interpolation operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_POINTS = 32


def _require_distinct(nodes: tuple[str, ...]) -> None:
    """Reject repeated interpolation nodes."""
    seen: set[str] = set()
    for n in nodes:
        if n in seen:
            raise ValueError("interpolation nodes must be pairwise distinct")
        seen.add(n)


class DividedDifferencesRequest(StrictModel):
    """Compute divided differences from sample points."""

    nodes: tuple[str, ...] = Field(min_length=1, max_length=MAX_POINTS)
    values: tuple[str, ...] = Field(min_length=1, max_length=MAX_POINTS)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if len(self.nodes) != len(self.values):
            raise ValueError("nodes and values must have the same length")
        _require_distinct(self.nodes)
        return self


class NewtonFormRequest(StrictModel):
    """Compute Newton form coefficients from divided differences."""

    nodes: tuple[str, ...] = Field(min_length=1, max_length=MAX_POINTS)
    values: tuple[str, ...] = Field(min_length=1, max_length=MAX_POINTS)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if len(self.nodes) != len(self.values):
            raise ValueError("nodes and values must have the same length")
        _require_distinct(self.nodes)
        return self


class NewtonEvaluateRequest(StrictModel):
    """Evaluate a polynomial in Newton form at a point."""

    nodes: tuple[str, ...] = Field(min_length=1, max_length=MAX_POINTS)
    values: tuple[str, ...] = Field(min_length=1, max_length=MAX_POINTS)
    evaluation_point: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if len(self.nodes) != len(self.values):
            raise ValueError("nodes and values must have the same length")
        _require_distinct(self.nodes)
        return self


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
