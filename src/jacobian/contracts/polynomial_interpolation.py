"""Typed wire contracts for exact polynomial interpolation operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian.contracts.base import ContractModel
from jacobian.contracts.exact import CanonicalRational

MAX_POINTS = 32


class RationalPoint(ContractModel):
    """One (x, y) data point for interpolation."""

    x: CanonicalRational
    y: CanonicalRational


class NewtonInterpolationRequest(ContractModel):
    """Newton-form interpolation through given data points."""

    points: tuple[RationalPoint, ...] = Field(min_length=1, max_length=MAX_POINTS)

    @model_validator(mode="after")
    def require_unique_x(self) -> Self:
        xs = [p.x.as_fraction() for p in self.points]
        if len(set(xs)) != len(xs):
            raise ValueError("interpolation x-values must be distinct")
        return self


class MultipointEvaluationRequest(ContractModel):
    """Evaluate a polynomial at multiple points simultaneously."""

    coefficients: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_POINTS + 1
    )
    evaluation_points: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_POINTS
    )


class NewtonInterpolationResult(ContractModel):
    """The interpolating polynomial in coefficient form [a_0, ..., a_n]."""

    coefficients: tuple[CanonicalRational, ...] = Field(min_length=1)
    divided_differences: tuple[CanonicalRational, ...] = Field(min_length=1)
    method: Literal["NEWTON_DIVIDED_DIFFERENCE"] = "NEWTON_DIVIDED_DIFFERENCE"


class MultipointEvaluationResult(ContractModel):
    """Polynomial values at the evaluation points."""

    values: tuple[CanonicalRational, ...] = Field(min_length=1)
    method: Literal["HORNER_EVALUATION"] = "HORNER_EVALUATION"


__all__ = [
    "MultipointEvaluationRequest",
    "MultipointEvaluationResult",
    "NewtonInterpolationRequest",
    "NewtonInterpolationResult",
    "RationalPoint",
]
