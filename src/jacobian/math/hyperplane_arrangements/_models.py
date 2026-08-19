"""Typed wire contracts for hyperplane arrangement operations."""

from __future__ import annotations

from typing import Self

import sympy
from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_HYPERPLANES = 16
MAX_DIM = 8


class RationalHyperplane(StrictModel):
    """A hyperplane {x : a . x = b} in R^n."""

    coefficients: tuple[str, ...] = Field(min_length=1, max_length=MAX_DIM)
    constant: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        try:
            parsed = [sympy.Rational(c) for c in self.coefficients]
        except (ValueError, TypeError, sympy.SympifyError) as exc:
            raise ValueError("coefficients must be exact rationals") from exc
        if all(c == 0 for c in parsed):
            raise ValueError("hyperplane coefficients must not all be zero")
        try:
            sympy.Rational(self.constant)
        except (ValueError, TypeError, sympy.SympifyError) as exc:
            raise ValueError("constant must be an exact rational") from exc
        return self


class HyperplaneArrangementRequest(StrictModel):
    """A central hyperplane arrangement in R^n."""

    ambient_dimension: int = Field(ge=1, le=MAX_DIM)
    hyperplanes: tuple[RationalHyperplane, ...] = Field(
        min_length=1, max_length=MAX_HYPERPLANES
    )

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        for hp in self.hyperplanes:
            if len(hp.coefficients) != self.ambient_dimension:
                raise ValueError("hyperplane coefficients must match ambient dimension")
        return self


class CharacteristicPolynomialRequest(StrictModel):
    ambient_dimension: int = Field(ge=1, le=MAX_DIM)
    hyperplane_count: int = Field(ge=1, le=MAX_HYPERPLANES)


class ChamberCountRequest(StrictModel):
    ambient_dimension: int = Field(ge=1, le=MAX_DIM)
    hyperplane_count: int = Field(ge=1, le=MAX_HYPERPLANES)


# Results


class HyperplaneArrangementResult(StrictModel):
    hyperplane_count: int = Field(ge=1)
    ambient_dimension: int = Field(ge=1)
    is_central: bool
    method: str = "ARRANGEMENT_CONSTRUCTION"


class CharacteristicPolynomialResult(StrictModel):
    coefficients: tuple[str, ...]
    degree: int = Field(ge=0)
    method: str = "SYMPY_CHARACTERISTIC_POLYNOMIAL"


class ChamberCountResult(StrictModel):
    chamber_count: int = Field(ge=1)
    method: str = "EXACT_FORMULA"
