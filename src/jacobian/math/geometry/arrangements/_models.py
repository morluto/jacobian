"""Typed wire contracts for hyperplane arrangement operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel

MAX_HYPERPLANES = 16
MAX_DIM = 8


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"hyperplane_arrangement.{reason}", message)


class RationalHyperplane(StrictModel):
    """A hyperplane {x : a . x = b} in R^n."""

    coefficients: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_DIM
    )
    constant: CanonicalRational

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if all(coefficient.as_fraction() == 0 for coefficient in self.coefficients):
            raise _validation_error(
                "coefficients_zero", "hyperplane coefficients must not all be zero"
            )
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
                raise _validation_error(
                    "dimension_mismatch",
                    "hyperplane coefficients must match ambient dimension",
                )
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


class CharacteristicPolynomialResult(StrictModel):
    coefficients: tuple[str, ...]
    degree: int = Field(ge=0)


class ChamberCountResult(StrictModel):
    chamber_count: int = Field(ge=1)
