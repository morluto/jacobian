"""Typed wire contracts for polynomial support geometry operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.polynomial_support_geometry.values import (
    MAX_NEWTON_DIMENSION,
    MAX_SUPPORT_TERMS,
    MAX_WEIGHT_COMPONENTS,
)

MAX_POLY_TERMS = 4096


class SupportRequest(StrictModel):
    """Request the support of a polynomial."""

    terms: tuple[dict, ...] = Field(min_length=1, max_length=MAX_POLY_TERMS)
    variables: tuple[str, ...] = Field(min_length=1, max_length=MAX_NEWTON_DIMENSION)


class NewtonPolytopeRequest(StrictModel):
    """Request the Newton polytope of a polynomial."""

    terms: tuple[dict, ...] = Field(min_length=1, max_length=MAX_POLY_TERMS)
    variables: tuple[str, ...] = Field(min_length=1, max_length=MAX_NEWTON_DIMENSION)


class WeightProfileRequest(StrictModel):
    """Request the weight profile of a polynomial."""

    terms: tuple[dict, ...] = Field(min_length=1, max_length=MAX_POLY_TERMS)
    variables: tuple[str, ...] = Field(min_length=1, max_length=MAX_NEWTON_DIMENSION)
    weight: tuple[int, ...] = Field(min_length=1, max_length=MAX_WEIGHT_COMPONENTS)

    @model_validator(mode="after")
    def require_matching_dimensions(self) -> Self:
        if len(self.weight) != len(self.variables):
            raise ValueError("weight vector length must match variable count")
        return self


class InitialFormRequest(StrictModel):
    """Request the initial form of a polynomial."""

    terms: tuple[dict, ...] = Field(min_length=1, max_length=MAX_POLY_TERMS)
    variables: tuple[str, ...] = Field(min_length=1, max_length=MAX_NEWTON_DIMENSION)
    weight: tuple[int, ...] = Field(min_length=1, max_length=MAX_WEIGHT_COMPONENTS)

    @model_validator(mode="after")
    def require_matching_dimensions(self) -> Self:
        if len(self.weight) != len(self.variables):
            raise ValueError("weight vector length must match variable count")
        return self


__all__ = [
    "InitialFormRequest",
    "NewtonPolytopeRequest",
    "SupportRequest",
    "WeightProfileRequest",
]
