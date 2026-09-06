"""Typed contracts for exact unit-circle polynomial operations."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.number_theory.number_fields.values import (
    RealNumberFieldEmbeddingRecord,
    SimpleNumberFieldElement,
    SimpleNumberFieldRealEmbeddingBinding,
)
from jacobian.math.polynomials.values import RationalPolynomial

MAX_ARC_ENERGY_DEGREE = 32
MAX_ARC_ENERGY_TERMS = 64
MAX_ARC_ENERGY_CONDUCTOR = 32
MAX_ARC_ENERGY_FIELD_DEGREE = 8
MAX_ARC_ENERGY_FIELD_COEFFICIENT_DIGITS = 256
# With one shared coefficient denominator D, these bounds give D < 10^48
# and scaled integer coefficients below 10^96.  A degree-32 correlation,
# division by lcm(1,...,32), and the largest fixed cyclotomic coordinate (14)
# therefore stay below 214 decimal digits, inside the 256-digit result carrier.
MAX_ARC_ENERGY_INPUT_COMPONENT_DIGITS = 48
MAX_ARC_ENERGY_TOTAL_DENOMINATOR_DIGITS = 48
MAX_FEJER_RIESZ_COMPONENT_DIGITS = 64
MAX_FEJER_RIESZ_DERIVED_DIGITS = 256


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"polynomial.unit_circle.{reason}", message)


class UnitCircleArcEnergyRequest(StrictModel):
    """A rational polynomial and an oriented unwrapped rational-turn arc."""

    polynomial: RationalPolynomial
    start_turn: CanonicalRational
    end_turn: CanonicalRational


class UnitCircleArcEnergyResult(StrictModel):
    """The exact energy ``A + B/pi`` in the arc's standard real cyclotomic field.

    The conductor is ``lcm(4, den(start), den(end))``.  The retained embedding
    sends the field generator to ``2*cos(2*pi/conductor)``.
    """

    polynomial: RationalPolynomial
    start_turn: CanonicalRational
    end_turn: CanonicalRational
    cyclotomic_conductor: StrictInt = Field(ge=4, le=MAX_ARC_ENERGY_CONDUCTOR)
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
    def require_canonical_terms(self) -> Self:
        exponents = tuple(term.exponent for term in self.terms)
        if len(set(exponents)) != len(exponents):
            raise _validation_error(
                "duplicate_laurent_exponents",
                "Laurent exponents must be unique",
            )
        if exponents != tuple(sorted(exponents)):
            raise _validation_error(
                "laurent_term_order",
                "Laurent terms must be ordered by increasing exponent",
            )
        if any(term.coefficient.as_fraction() == 0 for term in self.terms):
            raise _validation_error(
                "zero_laurent_term",
                "zero Laurent coefficients must be omitted",
            )
        return self


class RealDegreeOnePolynomialFactor(StrictModel):
    """A real degree-at-most-one factor in one exact embedded field."""

    embedding_record: RealNumberFieldEmbeddingRecord
    coefficients_ascending: tuple[SimpleNumberFieldElement, SimpleNumberFieldElement]
    variable: Literal["z"] = "z"

    @model_validator(mode="after")
    def bind_coefficients_to_field(self) -> Self:
        presentation = self.embedding_record.embedding.presentation
        if any(
            coefficient.presentation != presentation
            for coefficient in self.coefficients_ascending
        ):
            raise _validation_error(
                "factor_field",
                "factor coefficients must belong to the retained embedded field",
            )
        return self


class FejerRieszFactored(StrictModel):
    """The unique outer factor normalized by a positive constant term."""

    status: Literal["FACTORED"] = "FACTORED"
    factor: RealDegreeOnePolynomialFactor


class FejerRieszZero(StrictModel):
    """The identically zero Laurent polynomial."""

    status: Literal["ZERO"] = "ZERO"


class FejerRieszNegative(StrictModel):
    """An exact cosine witness where the Laurent polynomial is negative."""

    status: Literal["NEGATIVE"] = "NEGATIVE"
    cosine_witness: CanonicalRational


type FejerRieszConclusion = Annotated[
    FejerRieszFactored | FejerRieszZero | FejerRieszNegative,
    Field(discriminator="status"),
]


class FejerRieszFactorResult(StrictModel):
    """The exact degree-one Fejer-Riesz conclusion bound to its source."""

    source: HermitianLaurentPolynomial
    conclusion: FejerRieszConclusion


__all__ = [
    "FejerRieszFactorResult",
    "FejerRieszFactored",
    "FejerRieszNegative",
    "FejerRieszZero",
    "HermitianLaurentPolynomial",
    "HermitianLaurentTerm",
    "RealDegreeOnePolynomialFactor",
    "UnitCircleArcEnergyRequest",
    "UnitCircleArcEnergyResult",
]
