"""Typed contracts for exact unit-circle polynomial operations."""

from __future__ import annotations

from typing import Literal

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


__all__ = [
    "UnitCircleArcEnergyRequest",
    "UnitCircleArcEnergyResult",
]
