"""Coordinate composition and its retained pre-cancellation open locus."""

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.canonical import encode_strict_json
from jacobian.math.polynomials.rational_functions.values import RationalFunctionMap
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    require_sparse_polynomial_budget,
)


def guard_key(guard: RationalPolynomial) -> bytes:
    """The existing canonical polynomial encoding determines guard order."""
    return encode_strict_json(guard.model_dump(mode="json"))


class RationalMapCompositionRequest(StrictModel):
    outer: RationalFunctionMap
    inner: RationalFunctionMap

    @model_validator(mode="after")
    def require_intermediate_axis(self) -> Self:
        if self.inner.target_coordinates != self.outer.source_variables:
            raise ValueError(
                "inner target coordinates must equal outer source variables"
            )
        return self


class RationalFunctionMapComposition(StrictModel):
    """The composite field map with the domain of the supplied construction.

    Guards mean a conjunction of polynomial nonvanishing conditions. They
    retain every inner denominator and each substituted outer denominator,
    even when the canonical composite extends across a removed point.
    """

    outer: RationalFunctionMap
    inner: RationalFunctionMap
    composite: RationalFunctionMap
    construction_locus_guard: tuple[RationalPolynomial, ...] = Field(max_length=4104)

    @model_validator(mode="after")
    def require_axes_and_canonical_guards(self) -> Self:
        if self.inner.target_coordinates != self.outer.source_variables:
            raise ValueError(
                "inner target coordinates must equal outer source variables"
            )
        if (
            self.composite.source_variables != self.inner.source_variables
            or self.composite.target_coordinates != self.outer.target_coordinates
        ):
            raise ValueError(
                "composite must retain the inner source and outer target axes"
            )
        keys = []
        for guard in self.construction_locus_guard:
            if guard.variables != self.composite.source_variables:
                raise ValueError(
                    "construction guards must retain the composite source axis"
                )
            terms = guard.polynomial.terms
            if (
                not terms
                or terms[0].coefficient.as_fraction() != 1
                or not any(any(term.exponents) for term in terms)
            ):
                raise ValueError(
                    "construction guards must be nonconstant monic polynomials"
                )
            require_sparse_polynomial_budget(
                guard.polynomial,
                maximum_terms=256,
                maximum_exponent=64,
                maximum_coefficient_digits=128,
                label="construction guard",
            )
            keys.append(guard_key(guard))
        if keys != sorted(set(keys)):
            raise ValueError(
                "construction guards must use unique canonical encoding order"
            )
        return self
