"""Typed contracts for exact root--critical-point distance profiles.

The current carrier is deliberately a small, self-contained algebraic-root
carrier.  It keeps the exact indexed root identity and rational rectangle
needed by the profile while the general normal/splitting-field values are
being developed by the number-field owner.
"""

from __future__ import annotations

from math import gcd
from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, DecimalIntegerEncoding
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer
from jacobian.math.number_theory.algebraic_numbers.real import (
    RationalIsolatingInterval,
    RealAlgebraicValue,
)
from jacobian.math.polynomials.values import RationalPolynomial

MAX_ROOT_CRITICAL_DEGREE = 8
MAX_ROOT_CRITICAL_DISTANCE_DEGREE = 16
MAX_ROOT_CRITICAL_PAIRS = 64
MAX_ROOT_CRITICAL_ROOT_COMPONENT_DIGITS = 256
RootCriticalFactorCoefficient = Annotated[
    int, DecimalIntegerEncoding(max_digits=MAX_ROOT_CRITICAL_ROOT_COMPONENT_DIGITS)
]


def _error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"polynomial.root_critical.{reason}", message)


class RootCriticalRectangle(StrictModel):
    """A closed rational rectangle containing one selected complex root."""

    real_lower: CanonicalRational
    real_upper: CanonicalRational
    imaginary_lower: CanonicalRational
    imaginary_upper: CanonicalRational

    @model_validator(mode="after")
    def require_ordered_bounds(self) -> Self:
        if self.real_lower.as_fraction() > self.real_upper.as_fraction():
            raise _error("rectangle_order", "real rectangle bounds are reversed")
        if self.imaginary_lower.as_fraction() > self.imaginary_upper.as_fraction():
            raise _error("rectangle_order", "imaginary rectangle bounds are reversed")
        for component in (
            self.real_lower,
            self.real_upper,
            self.imaginary_lower,
            self.imaginary_upper,
        ):
            if (
                max(
                    len(format_canonical_integer(abs(component.num))),
                    len(format_canonical_integer(component.den)),
                )
                > MAX_ROOT_CRITICAL_ROOT_COMPONENT_DIGITS
            ):
                raise _error(
                    "rectangle_digits",
                    "root rectangle components exceed the bounded exact carrier",
                )
        return self


class RootCriticalRoot(StrictModel):
    """One distinct source or derivative root with source multiplicity."""

    axis_index: StrictInt = Field(ge=0, le=MAX_ROOT_CRITICAL_DEGREE - 1)
    factor: tuple[RootCriticalFactorCoefficient, ...] = Field(
        min_length=2, max_length=MAX_ROOT_CRITICAL_DEGREE + 1
    )
    root_index: StrictInt = Field(ge=0, le=MAX_ROOT_CRITICAL_DEGREE - 1)
    multiplicity: StrictInt = Field(ge=1, le=MAX_ROOT_CRITICAL_DEGREE)
    rectangle: RootCriticalRectangle

    @model_validator(mode="after")
    def require_canonical_factor(self) -> Self:
        if self.factor[0] <= 0:
            raise _error(
                "factor_leading_sign", "root factors need positive leading coefficient"
            )
        content = 0
        for coefficient in self.factor:
            content = gcd(content, abs(coefficient))
        if content != 1:
            raise _error("factor_content", "root factors must be primitive")
        if self.root_index >= len(self.factor) - 1:
            raise _error("factor_root_index", "root index exceeds factor degree")
        return self


class RootCriticalDistanceRow(StrictModel):
    """One complete ordered pair and its exact nonnegative squared distance."""

    root_axis_index: StrictInt = Field(ge=0, le=MAX_ROOT_CRITICAL_DEGREE - 1)
    critical_axis_index: StrictInt = Field(ge=0, le=MAX_ROOT_CRITICAL_DEGREE - 1)
    distance_squared: RealAlgebraicValue
    isolating_interval: RationalIsolatingInterval
    kind: Literal["POSITIVE", "ZERO_DISTANCE"]

    @model_validator(mode="after")
    def require_real_distance_shape(self) -> Self:
        if (
            len(self.distance_squared.polynomial) - 1
            > MAX_ROOT_CRITICAL_DISTANCE_DEGREE
        ):
            raise _error(
                "distance_degree",
                "distance algebraic degree exceeds the admitted exact carrier",
            )
        if self.isolating_interval.lower.as_fraction() < 0:
            raise _error("distance_sign", "distance enclosure must be nonnegative")
        if self.kind == "ZERO_DISTANCE":
            if self.distance_squared.polynomial != (1, 0):
                raise _error(
                    "zero_distance", "zero rows must carry the canonical zero value"
                )
            if (
                self.isolating_interval.lower.as_fraction() != 0
                or self.isolating_interval.upper.as_fraction() != 0
            ):
                raise _error(
                    "zero_distance", "zero rows need a singleton zero interval"
                )
        return self


class RootCriticalDistanceProfileRequest(StrictModel):
    """One nonconstant univariate rational polynomial and semantic limits."""

    polynomial: RationalPolynomial
    max_pair_rows: StrictInt = Field(
        default=MAX_ROOT_CRITICAL_PAIRS, ge=0, le=MAX_ROOT_CRITICAL_PAIRS
    )


class RootCriticalDistanceProfile(StrictModel):
    """Complete source-bound root/critical family and Cartesian distance axis."""

    source_polynomial: RationalPolynomial
    derivative: RationalPolynomial
    roots: tuple[RootCriticalRoot, ...] = Field(max_length=MAX_ROOT_CRITICAL_DEGREE)
    critical_points: tuple[RootCriticalRoot, ...] = Field(
        max_length=MAX_ROOT_CRITICAL_DEGREE
    )
    pairs: tuple[RootCriticalDistanceRow, ...] = Field(
        max_length=MAX_ROOT_CRITICAL_PAIRS
    )
    representation: Literal["EXACT_REAL_ALGEBRAIC_SQUARED_DISTANCE_V1"] = (
        "EXACT_REAL_ALGEBRAIC_SQUARED_DISTANCE_V1"
    )

    @model_validator(mode="after")
    def require_complete_axes(self) -> Self:
        if len(self.source_polynomial.variables) != 1:
            raise _error("univariate", "root-critical profiles require one variable")
        if self.derivative.variables != self.source_polynomial.variables:
            raise _error(
                "derivative_axis", "derivative must use the source variable axis"
            )
        degree = max(
            (term.exponents[0] for term in self.source_polynomial.polynomial.terms),
            default=0,
        )
        derivative_degree = max(
            (term.exponents[0] for term in self.derivative.polynomial.terms),
            default=0,
        )
        if degree <= 0:
            raise _error(
                "constant", "constant polynomials have no root-critical profile"
            )
        if sum(root.multiplicity for root in self.roots) != degree:
            raise _error(
                "root_multiplicity",
                "root multiplicities must reconstruct source degree",
            )
        if sum(root.multiplicity for root in self.critical_points) != derivative_degree:
            raise _error(
                "critical_multiplicity",
                "critical multiplicities must reconstruct derivative degree",
            )
        if tuple(root.axis_index for root in self.roots) != tuple(
            range(len(self.roots))
        ):
            raise _error("root_axis", "root axis indices must be contiguous")
        if tuple(root.axis_index for root in self.critical_points) != tuple(
            range(len(self.critical_points))
        ):
            raise _error("critical_axis", "critical axis indices must be contiguous")
        expected = {
            (root.axis_index, critical.axis_index)
            for root in self.roots
            for critical in self.critical_points
        }
        actual = {(row.root_axis_index, row.critical_axis_index) for row in self.pairs}
        if actual != expected or len(self.pairs) != len(expected):
            raise _error(
                "pair_completeness",
                "pairs must be the complete rectangular root-critical axis",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        source_polynomial: RationalPolynomial,
        derivative: RationalPolynomial,
        roots: tuple[RootCriticalRoot, ...],
        critical_points: tuple[RootCriticalRoot, ...],
        pairs: tuple[RootCriticalDistanceRow, ...],
    ) -> Self:
        return cls.model_construct(
            source_polynomial=source_polynomial,
            derivative=derivative,
            roots=roots,
            critical_points=critical_points,
            pairs=pairs,
            representation="EXACT_REAL_ALGEBRAIC_SQUARED_DISTANCE_V1",
        )


__all__ = [
    "MAX_ROOT_CRITICAL_DEGREE",
    "MAX_ROOT_CRITICAL_DISTANCE_DEGREE",
    "MAX_ROOT_CRITICAL_PAIRS",
    "RootCriticalDistanceProfile",
    "RootCriticalDistanceProfileRequest",
    "RootCriticalDistanceRow",
    "RootCriticalRectangle",
    "RootCriticalRoot",
]
