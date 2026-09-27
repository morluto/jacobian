"""Canonical exact windows on a rational Puiseux exponent lattice."""

from __future__ import annotations

from math import lcm
from typing import Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel
from jacobian.math.polynomials.local_series.values import (
    MAX_LOCAL_SERIES_COEFFICIENT_DIGITS,
    MAX_LOCAL_SERIES_EXPONENT,
    MAX_LOCAL_SERIES_TERMS,
)
from jacobian.math.polynomials.values import PolynomialVariable

MAX_PUISEUX_RAMIFICATION = 256


def _err(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"local_series.{reason}", message)


class PuiseuxTerm(StrictModel):
    """One nonzero coefficient at an exact rational exponent."""

    exponent: CanonicalRational
    coefficient: CanonicalRational

    @model_validator(mode="after")
    def validate_term(self) -> Self:
        for field, value in (
            ("exponent", self.exponent),
            ("coefficient", self.coefficient),
        ):
            try:
                require_bounded_rational(
                    value,
                    max_digits=(
                        MAX_LOCAL_SERIES_EXPONENT
                        if field == "exponent"
                        else MAX_LOCAL_SERIES_COEFFICIENT_DIGITS
                    ),
                    label=f"Puiseux {field}",
                )
            except ValueError as error:
                raise _err("puiseux_bound", str(error)) from error
        if abs(self.exponent.as_fraction()) > MAX_LOCAL_SERIES_EXPONENT:
            raise _err(
                "puiseux_exponent_bound",
                "Puiseux term exponent exceeds the shared representation limit",
            )
        if self.coefficient.as_fraction() == 0:
            raise _err(
                "puiseux_zero_term", "Puiseux terms must have nonzero coefficients"
            )
        return self


class TruncatedPuiseuxWindow(StrictModel):
    """A finite exact Puiseux prefix with a rational Big-O cutoff.

    It represents ``sum(c_q * (x-center)^q) + O((x-center)^precision)``.
    Missing exponents on the declared ramification lattice have known zero
    coefficients; exponents at or above ``precision`` are unknown.
    """

    variable: PolynomialVariable = "t"
    center: CanonicalRational = Field(default=CanonicalRational(num=0, den=1))
    valuation_lower: CanonicalRational
    precision: CanonicalRational
    ramification_index: StrictInt = Field(ge=1, le=MAX_PUISEUX_RAMIFICATION)
    terms: tuple[PuiseuxTerm, ...] = Field(max_length=MAX_LOCAL_SERIES_TERMS)

    @model_validator(mode="after")
    def validate_window(self) -> Self:
        for label, value in (
            ("lower exponent", self.valuation_lower),
            ("precision exponent", self.precision),
            ("center", self.center),
        ):
            try:
                require_bounded_rational(
                    value,
                    max_digits=MAX_LOCAL_SERIES_COEFFICIENT_DIGITS,
                    label=f"Puiseux {label}",
                )
            except ValueError as error:
                raise _err("puiseux_bound", str(error)) from error
        lower = self.valuation_lower.as_fraction()
        precision = self.precision.as_fraction()
        if (
            abs(lower) > MAX_LOCAL_SERIES_EXPONENT
            or abs(precision) > MAX_LOCAL_SERIES_EXPONENT
        ):
            raise _err(
                "puiseux_exponent_bound",
                "Puiseux window exponents exceed the shared representation limit",
            )
        if lower > precision:
            raise _err("puiseux_window", "Puiseux window requires lower <= precision")
        rationals = [lower, precision]
        prior = None
        for term in self.terms:
            exponent = term.exponent.as_fraction()
            if not lower <= exponent < precision:
                raise _err(
                    "puiseux_term_range",
                    "Puiseux term lies outside the retained exponent window",
                )
            if prior is not None and exponent <= prior:
                raise _err(
                    "puiseux_term_order",
                    "Puiseux terms must have distinct increasing exponents",
                )
            prior = exponent
            rationals.append(exponent)
        required = lcm(*(value.denominator for value in rationals))
        if required > MAX_PUISEUX_RAMIFICATION:
            raise _err(
                "puiseux_ramification_bound",
                "Puiseux exponent lattice exceeds the ramification bound",
            )
        if self.ramification_index != required:
            raise _err(
                "puiseux_ramification_mismatch",
                "ramification_index must be the least common denominator of the window and retained exponents",
            )
        return self


__all__ = ["MAX_PUISEUX_RAMIFICATION", "PuiseuxTerm", "TruncatedPuiseuxWindow"]
