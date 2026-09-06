"""Typed contracts for exact unit-circle polynomial operations."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.number_theory.number_fields.values import (
    SimpleNumberFieldRealEmbeddingBinding,
)
from jacobian.math.polynomials.values import RationalPolynomial

MAX_ARC_ENERGY_DEGREE = 32
MAX_ARC_ENERGY_TERMS = 64
MAX_ARC_ENERGY_CONDUCTOR = 32
MAX_ARC_ENERGY_FIELD_DEGREE = 8
MAX_ARC_ENERGY_FIELD_COEFFICIENT_DIGITS = 256


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"polynomial.unit_circle.{reason}", message)


class UnitCircleArcEnergyRequest(StrictModel):
    """A rational polynomial and an oriented unwrapped rational-turn arc."""

    polynomial: RationalPolynomial
    start_turn: CanonicalRational
    end_turn: CanonicalRational


class UnitCircleArcEnergyResult(StrictModel):
    """The exact normalized arc energy ``A + B/pi`` with its source."""

    polynomial: RationalPolynomial
    start_turn: CanonicalRational
    end_turn: CanonicalRational
    rational_part: CanonicalRational
    pi_inverse_coefficient: SimpleNumberFieldRealEmbeddingBinding
    representation: Literal["RATIONAL_PLUS_REAL_CYCLOTOMIC_OVER_PI"] = (
        "RATIONAL_PLUS_REAL_CYCLOTOMIC_OVER_PI"
    )


class HermitianLaurentTerm(StrictModel):
    """One rational coefficient of a scalar Hermitian Laurent polynomial."""

    exponent: StrictInt = Field(ge=-1, le=1)
    coefficient: CanonicalRational


class HermitianLaurentPolynomial(StrictModel):
    """The bounded degree-one rational Hermitian Laurent input slice."""

    terms: tuple[HermitianLaurentTerm, ...] = Field(max_length=3)

    @model_validator(mode="after")
    def require_unique_exponents(self) -> HermitianLaurentPolynomial:
        exponents = tuple(term.exponent for term in self.terms)
        if len(set(exponents)) != len(exponents):
            raise _validation_error(
                "duplicate_laurent_exponents",
                "Laurent exponents must be unique",
            )
        return self


class FejerRieszFactorResult(StrictModel):
    """A normalized scalar Fejer-Riesz factor and its Laurent source."""

    source: HermitianLaurentPolynomial
    factor_coefficients: tuple[SimpleNumberFieldRealEmbeddingBinding, ...] = Field(
        min_length=1, max_length=2
    )
    field_degree: int = Field(ge=1, le=MAX_ARC_ENERGY_FIELD_DEGREE)
    zero_input: bool
    normalization: Literal["OUTER_Q0_POSITIVE_REAL"] = "OUTER_Q0_POSITIVE_REAL"


__all__ = [
    "FejerRieszFactorResult",
    "HermitianLaurentPolynomial",
    "HermitianLaurentTerm",
    "UnitCircleArcEnergyRequest",
    "UnitCircleArcEnergyResult",
]
