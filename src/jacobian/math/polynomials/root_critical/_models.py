"""Typed contracts for exact root--critical-point distance profiles.

Source and critical roots reuse the domain-owned real and complex algebraic
carriers. Multiplicity and isolating rectangles remain profile metadata.
"""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer
from jacobian.math.number_theory.algebraic_numbers.complex import (
    ComplexAlgebraicValue,
)
from jacobian.math.number_theory.algebraic_numbers.real import (
    RationalIsolatingInterval,
    RealAlgebraicValue,
)
from jacobian.math.polynomials.values import RationalPolynomial

MAX_ROOT_CRITICAL_DEGREE = 8
MAX_ROOT_CRITICAL_DISTANCE_DEGREE = 16
MAX_ROOT_CRITICAL_PAIRS = 64
MAX_ROOT_CRITICAL_ROOT_COMPONENT_DIGITS = 256
MAX_SPLITTING_FIELD_DEGREE = 16
MAX_SPLITTING_FIELD_CELLS = 64


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
    value: RealAlgebraicValue | ComplexAlgebraicValue
    multiplicity: StrictInt = Field(ge=1, le=MAX_ROOT_CRITICAL_DEGREE)
    rectangle: RootCriticalRectangle


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


class ExactSplittingFieldRequest(StrictModel):
    """Request the exact splitting field of one bounded rational polynomial."""

    polynomial: RationalPolynomial
    embedding_index: StrictInt = Field(default=0, ge=0, le=MAX_SPLITTING_FIELD_CELLS)


class SplittingFieldRoot(StrictModel):
    """One distinct root as an exact primitive-element polynomial."""

    axis_index: StrictInt = Field(ge=0, le=MAX_SPLITTING_FIELD_CELLS)
    coefficients_ascending: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_SPLITTING_FIELD_DEGREE
    )
    multiplicity: StrictInt = Field(ge=1, le=MAX_ROOT_CRITICAL_DEGREE)
    rectangle: RootCriticalRectangle


class ExactSplittingField(StrictModel):
    """A complete bounded exact splitting field of the square-free p*p' support.

    The field is presented by one primitive element theta with its monic
    minimal polynomial over QQ; every distinct root is an exact rational
    polynomial in theta. ``conjugation_coefficients`` is the image of theta
    under complex conjugation, so conjugation is an exact Q-algebra
    automorphism of the retained field.
    """

    source_polynomial: RationalPolynomial
    squarefree_support: RationalPolynomial
    defining_polynomial: RationalPolynomial
    conjugation_coefficients: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_SPLITTING_FIELD_DEGREE
    )
    embedding_index: StrictInt = Field(ge=0, le=MAX_SPLITTING_FIELD_CELLS)
    embedding_rectangle: RootCriticalRectangle
    roots: tuple[SplittingFieldRoot, ...] = Field(max_length=MAX_SPLITTING_FIELD_CELLS)

    @model_validator(mode="after")
    def require_field_shape(self) -> Self:
        if self.source_polynomial.variables != self.squarefree_support.variables:
            raise _error("field_source_axis", "field supports must share one axis")
        if len(self.defining_polynomial.variables) != 1:
            raise _error(
                "field_defining_axis",
                "the defining polynomial of the primitive element must be univariate",
            )
        degree = (
            self.defining_polynomial.polynomial.terms[0].exponents[0]
            if self.defining_polynomial.polynomial.terms
            else 0
        )
        if not 1 <= degree <= MAX_SPLITTING_FIELD_DEGREE:
            raise _error("field_degree", "the splitting-field degree exceeds its bound")
        if len(self.conjugation_coefficients) != degree:
            raise _error(
                "field_conjugation_degree",
                "the conjugation element must have one coefficient per theta power",
            )
        for root in self.roots:
            if len(root.coefficients_ascending) != degree:
                raise _error(
                    "field_root_degree",
                    "every root must have one coefficient per theta power",
                )
        if tuple(root.axis_index for root in self.roots) != tuple(
            range(len(self.roots))
        ):
            raise _error("field_root_axis", "field root indices must be contiguous")
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


class SplittingFieldDistanceRequest(StrictModel):
    """Bind the root-critical distance profile to a supplied exact splitting field."""

    polynomial: RationalPolynomial
    splitting_field: ExactSplittingField
    max_pair_rows: StrictInt = Field(
        default=MAX_ROOT_CRITICAL_PAIRS, ge=0, le=MAX_ROOT_CRITICAL_PAIRS
    )


class SplittingFieldDistanceProfile(StrictModel):
    """Exact root-critical distances computed inside the supplied splitting field."""

    splitting_field: ExactSplittingField
    profile: RootCriticalDistanceProfile
    root_field_coefficients: tuple[tuple[CanonicalRational, ...], ...] = Field(
        max_length=MAX_ROOT_CRITICAL_DEGREE
    )
    critical_field_coefficients: tuple[tuple[CanonicalRational, ...], ...] = Field(
        max_length=MAX_ROOT_CRITICAL_DEGREE
    )
    distance_field_coefficients: tuple[tuple[CanonicalRational, ...], ...] = Field(
        max_length=MAX_ROOT_CRITICAL_PAIRS
    )

    @model_validator(mode="after")
    def require_bound_axes(self) -> Self:
        if self.splitting_field.source_polynomial != self.profile.source_polynomial:
            raise _error(
                "binding_source_mismatch",
                "the splitting field must be bound to the exact source polynomial",
            )
        if len(self.root_field_coefficients) != len(self.profile.roots):
            raise _error(
                "binding_root_axis", "root field rows must match the root axis"
            )
        if len(self.critical_field_coefficients) != len(self.profile.critical_points):
            raise _error(
                "binding_critical_axis",
                "critical field rows must match the critical axis",
            )
        if len(self.distance_field_coefficients) != len(self.profile.pairs):
            raise _error("binding_pair_axis", "distance rows must match the pair axis")
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


class SplittingFieldKernelPayload(StrictModel):
    """Strict decoder for one bounded splitting-field worker response."""

    squarefree_support: RationalPolynomial
    defining_polynomial: RationalPolynomial
    conjugation_coefficients: tuple[CanonicalRational, ...]
    embedding_index: StrictInt
    embedding_rectangle: RootCriticalRectangle
    roots: tuple[SplittingFieldRoot, ...]
    root_field_coefficients: tuple[tuple[CanonicalRational, ...], ...] = ()
    critical_field_coefficients: tuple[tuple[CanonicalRational, ...], ...] = ()
    distance_field_coefficients: tuple[tuple[CanonicalRational, ...], ...] = ()


__all__ = [
    "MAX_ROOT_CRITICAL_DEGREE",
    "MAX_ROOT_CRITICAL_DISTANCE_DEGREE",
    "MAX_ROOT_CRITICAL_PAIRS",
    "MAX_SPLITTING_FIELD_CELLS",
    "MAX_SPLITTING_FIELD_DEGREE",
    "ExactSplittingField",
    "ExactSplittingFieldRequest",
    "RootCriticalDistanceProfile",
    "RootCriticalDistanceProfileRequest",
    "RootCriticalDistanceRow",
    "RootCriticalRectangle",
    "RootCriticalRoot",
    "SplittingFieldDistanceProfile",
    "SplittingFieldDistanceRequest",
    "SplittingFieldKernelPayload",
    "SplittingFieldRoot",
]
