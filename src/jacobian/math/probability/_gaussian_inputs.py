"""Boundary parser for canonical Gaussian sparse-polynomial requests."""

from __future__ import annotations

from fractions import Fraction
from typing import Annotated, Self

from pydantic import BeforeValidator, Field, StrictInt, ValidationError, model_validator

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer
from jacobian.math.probability._models import (
    MAX_GAUSSIAN_POLYNOMIAL_TERMS,
    MAX_GAUSSIAN_TERM_DEGREE,
    MAX_GAUSSIAN_VARIABLES,
    MAX_INPUT_RATIONAL_DIGITS,
    ExactComplexRational,
    GaussianPolynomial,
    GaussianPolynomialMomentRequest,
    GaussianPolynomialTerm,
)


class RawGaussianPolynomialTerm(StrictModel):
    """One bounded wire term before duplicate/zero canonicalization."""

    coefficient: ExactComplexRational
    exponents: tuple[StrictInt, ...] = Field(
        min_length=1,
        max_length=MAX_GAUSSIAN_VARIABLES,
    )

    @model_validator(mode="after")
    def require_bounded_term(self) -> Self:
        if any(
            type(exponent) is not int or exponent < 0 for exponent in self.exponents
        ):
            raise ValueError(
                "Gaussian polynomial exponents must be nonnegative integers"
            )
        if sum(self.exponents) > MAX_GAUSSIAN_TERM_DEGREE:
            raise ValueError(
                "Gaussian polynomial term exceeds the "
                f"{MAX_GAUSSIAN_TERM_DEGREE}-degree bound"
            )
        for component in (self.coefficient.real, self.coefficient.imaginary):
            require_bounded_rational(
                component,
                max_digits=MAX_INPUT_RATIONAL_DIGITS,
                label="Gaussian polynomial input coefficient",
            )
        return self


class RawGaussianPolynomial(StrictModel):
    """Bounded public representation accepted only at the request boundary."""

    variable_count: StrictInt = Field(ge=1, le=MAX_GAUSSIAN_VARIABLES)
    terms: tuple[RawGaussianPolynomialTerm, ...] = Field(
        min_length=1,
        max_length=MAX_GAUSSIAN_POLYNOMIAL_TERMS,
    )

    @model_validator(mode="after")
    def require_consistent_dimension(self) -> Self:
        if any(len(term.exponents) != self.variable_count for term in self.terms):
            raise ValueError(
                "every Gaussian polynomial exponent vector must match variable_count"
            )
        return self


def _rational(value: Fraction) -> CanonicalRational:
    return CanonicalRational(
        num=format_canonical_integer(value.numerator),
        den=format_canonical_integer(value.denominator),
    )


def canonical_gaussian_polynomial(value: object) -> GaussianPolynomial:
    """Parse a strict canonical value or canonicalize bounded loose wire terms."""

    if isinstance(value, GaussianPolynomial):
        return value

    # A canonical request is itself a valid public boundary form. Parsing it first
    # makes the canonical JSON projection replayable after duplicate raw terms have
    # legitimately accumulated beyond the per-raw-term 128-digit bound. The strict
    # value still enforces the 16-term and 4,096-digit component bounds.
    try:
        return GaussianPolynomial.model_validate(value)
    except ValidationError:
        pass

    raw = RawGaussianPolynomial.model_validate(value)
    combined: dict[tuple[int, ...], tuple[Fraction, Fraction]] = {}
    for term in raw.terms:
        real, imaginary = term.coefficient.as_fractions()
        previous_real, previous_imaginary = combined.get(
            term.exponents,
            (Fraction(), Fraction()),
        )
        combined[term.exponents] = (
            previous_real + real,
            previous_imaginary + imaginary,
        )

    terms = tuple(
        GaussianPolynomialTerm(
            coefficient=ExactComplexRational(
                real=_rational(real),
                imaginary=_rational(imaginary),
            ),
            exponents=exponents,
        )
        for exponents, (real, imaginary) in sorted(combined.items())
        if real or imaginary
    )
    if not terms:
        raise ValueError("Gaussian polynomial canonicalization removed every zero term")
    return GaussianPolynomial(variable_count=raw.variable_count, terms=terms)


class CanonicalGaussianPolynomialMomentRequest(GaussianPolynomialMomentRequest):
    """Request accepting strict canonical values or bounded loose wire terms."""

    polynomial: Annotated[
        GaussianPolynomial,
        BeforeValidator(
            canonical_gaussian_polynomial,
            json_schema_input_type=RawGaussianPolynomial | GaussianPolynomial,
        ),
    ]


__all__ = [
    "CanonicalGaussianPolynomialMomentRequest",
    "RawGaussianPolynomial",
    "RawGaussianPolynomialTerm",
    "canonical_gaussian_polynomial",
]
