"""Typed contracts for exact unit-circle supremum norms of polynomials.

The main result is intentionally a *certified rational enclosure* together
with the complete critical-point ledger, not a ``RealAlgebraicValue``.  The
maximum of ``|P(z)|^2`` on ``|z| = 1`` is the value of the rational function
``Q(t) / (1 + t**2)**degree`` at a real root of an exact derivative
numerator.  Its minimal polynomial can have degree exponential in the input
degree, far beyond the shared degree-16 real-algebraic carrier and its
degree-8 distinct-polynomial comparison envelope, so this owner does not
pretend to normalize it to a minimal polynomial.  Instead the exact value is
bound to its defining data (the retained numerator, denominator exponent, and
the isolating root index) and every reported quantity is a rational interval
that provably brackets it.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictBool, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.number_theory.algebraic_numbers.real import (
    RationalIsolatingInterval,
)
from jacobian.math.number_theory.number_fields import GaussianRational

# The transformed numerator has degree at most ``2 * degree`` and the exact
# derivative numerator at most ``2 * degree + 1``.  Exact real-root isolation
# is performed on that derivative numerator, so the number of real critical
# points is bounded by its degree.  The degree limit keeps the pairwise exact
# comparison ledger and the resultant-based value identification inside the
# measured envelope of the fixed SymPy backend.
MAX_SUP_NORM_DEGREE = 8
MAX_SUP_NORM_TERMS = MAX_SUP_NORM_DEGREE + 1
MAX_SUP_NORM_INPUT_COMPONENT_DIGITS = 32
MAX_SUP_NORM_NUMERATOR_DEGREE = 2 * MAX_SUP_NORM_DEGREE
MAX_SUP_NORM_DERIVATIVE_DEGREE = 2 * MAX_SUP_NORM_DEGREE + 1
MAX_SUP_NORM_CRITICAL_ROOTS = MAX_SUP_NORM_DERIVATIVE_DEGREE
MAX_SUP_NORM_DERIVATIVE_COMPONENT_DIGITS = 512
# One comparison ledger entry per distinct critical value plus the z=-1
# endpoint; this is the complete pairwise comparison budget.
MAX_SUP_NORM_COMPARISONS = (
    (MAX_SUP_NORM_CRITICAL_ROOTS + 1) * (MAX_SUP_NORM_CRITICAL_ROOTS + 2)
) // 2
MAX_SUP_NORM_ENCLOSURE_COMPONENT_DIGITS = 256
MAX_SUP_NORM_CERTIFICATE_BYTES = 262_144
# Deliberately conservative isolation/resultant work proxy over the fixed
# exact backend: ``(derivative_degree + 1)**3 * (derived_digits + 64)``.
MAX_SUP_NORM_ISOLATION_WORK = 3_000_000


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"polynomial.unit_circle.sup_norm_{reason}", message)


class GaussianRationalPolynomialTerm(StrictModel):
    """One nonzero Gaussian-rational monomial of the input polynomial."""

    coefficient: GaussianRational
    exponent: StrictInt = Field(ge=0, le=MAX_SUP_NORM_DEGREE)


class GaussianRationalPolynomial(StrictModel):
    """A bounded univariate polynomial over ``QQ(i)`` on the ``z`` axis.

    Terms are nonzero, unique, and listed in descending exponent order.
    The zero polynomial is the empty term list.
    """

    domain: Literal["QQ(i)"] = "QQ(i)"
    variable: Literal["z"] = "z"
    terms: tuple[GaussianRationalPolynomialTerm, ...] = Field(
        default=(),
        max_length=MAX_SUP_NORM_TERMS,
    )

    @model_validator(mode="after")
    def require_canonical_terms(self) -> Self:
        exponents = tuple(term.exponent for term in self.terms)
        if len(set(exponents)) != len(exponents):
            raise _validation_error(
                "duplicate_exponents", "polynomial exponents must be unique"
            )
        if exponents != tuple(sorted(exponents, reverse=True)):
            raise _validation_error(
                "term_order", "polynomial terms must use descending exponent order"
            )
        return self


class UnitCircleSupNormSquaredRequest(StrictModel):
    """One bounded Gaussian-rational polynomial in ``z``."""

    polynomial: GaussianRationalPolynomial


class UnitCircleCriticalPoint(StrictModel):
    """One finite real critical parameter and its exact value certificate.

    ``root_index`` is the zero-based index of this parameter among the
    increasing real roots of the retained ``derivative_numerator``.  The
    parameter and value intervals are certified rational enclosures.  The
    ``z_real``/``z_imaginary`` intervals bracket the image of the parameter
    interval under ``z = (1 + i t) / (1 - i t)``; they are singletons exactly
    when the critical parameter is rational.  ``comparison_rank`` counts the
    complete candidate family (finite critical values and the ``z = -1``
    endpoint) that is strictly greater, so rank zero identifies maximizers.
    """

    root_index: StrictInt = Field(ge=0, lt=MAX_SUP_NORM_CRITICAL_ROOTS)
    parameter: RationalIsolatingInterval
    value: RationalIsolatingInterval
    z_real: RationalIsolatingInterval
    z_imaginary: RationalIsolatingInterval
    comparison_rank: StrictInt = Field(ge=0, le=MAX_SUP_NORM_CRITICAL_ROOTS + 1)
    is_maximizer: StrictBool


class UnitCircleSupNormSquaredResult(StrictModel):
    """Certified unit-circle supremum data bound to its exact source.

    The result retains the exact transformed numerator ``Q`` and denominator
    exponent ``d`` so that ``|P(z)|^2 = Q(t) / (1 + t**2)**d`` (for
    ``z = (1 + i t) / (1 - i t)``) can be replayed, the exact derivative
    numerator whose real roots are the finite critical points, the complete
    critical ledger, and the separate ``z = -1`` endpoint candidate.
    """

    polynomial: GaussianRationalPolynomial
    degree: StrictInt = Field(ge=0, le=MAX_SUP_NORM_DEGREE)
    transformed_numerator: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_SUP_NORM_NUMERATOR_DEGREE + 1
    )
    denominator_exponent: StrictInt = Field(ge=0, le=MAX_SUP_NORM_DEGREE)
    derivative_numerator: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_SUP_NORM_DERIVATIVE_DEGREE + 1
    )
    critical_points: tuple[UnitCircleCriticalPoint, ...] = Field(
        default=(), max_length=MAX_SUP_NORM_CRITICAL_ROOTS
    )
    endpoint_minus_one_value: CanonicalRational
    endpoint_minus_one_is_maximizer: StrictBool
    maximizing_status: Literal[
        "FULL_CIRCLE",
        "ENDPOINT",
        "ISOLATED_CRITICAL",
        "CRITICAL_AND_ENDPOINT",
    ]
    sup_norm_squared_enclosure: RationalIsolatingInterval
    sup_norm_enclosure: RationalIsolatingInterval
    representation: Literal["CERTIFIED_RATIONAL_ENCLOSURE_WITH_CRITICAL_LEDGER"] = (
        "CERTIFIED_RATIONAL_ENCLOSURE_WITH_CRITICAL_LEDGER"
    )

    @classmethod
    def _from_kernel(
        cls,
        *,
        polynomial: GaussianRationalPolynomial,
        degree: int,
        transformed_numerator: tuple[CanonicalRational, ...],
        denominator_exponent: int,
        derivative_numerator: tuple[CanonicalRational, ...],
        critical_points: tuple[UnitCircleCriticalPoint, ...],
        endpoint_minus_one_value: CanonicalRational,
        endpoint_minus_one_is_maximizer: bool,
        maximizing_status: str,
        sup_norm_squared_enclosure: RationalIsolatingInterval,
        sup_norm_enclosure: RationalIsolatingInterval,
    ) -> Self:
        """Construct after the admitted exact kernel established every field."""

        return cls.model_construct(
            polynomial=polynomial,
            degree=degree,
            transformed_numerator=transformed_numerator,
            denominator_exponent=denominator_exponent,
            derivative_numerator=derivative_numerator,
            critical_points=critical_points,
            endpoint_minus_one_value=endpoint_minus_one_value,
            endpoint_minus_one_is_maximizer=endpoint_minus_one_is_maximizer,
            maximizing_status=maximizing_status,
            sup_norm_squared_enclosure=sup_norm_squared_enclosure,
            sup_norm_enclosure=sup_norm_enclosure,
        )


__all__ = [
    "MAX_SUP_NORM_CERTIFICATE_BYTES",
    "MAX_SUP_NORM_COMPARISONS",
    "MAX_SUP_NORM_CRITICAL_ROOTS",
    "MAX_SUP_NORM_DEGREE",
    "MAX_SUP_NORM_DERIVATIVE_COMPONENT_DIGITS",
    "MAX_SUP_NORM_DERIVATIVE_DEGREE",
    "MAX_SUP_NORM_ENCLOSURE_COMPONENT_DIGITS",
    "MAX_SUP_NORM_INPUT_COMPONENT_DIGITS",
    "MAX_SUP_NORM_ISOLATION_WORK",
    "MAX_SUP_NORM_NUMERATOR_DEGREE",
    "MAX_SUP_NORM_TERMS",
    "GaussianRationalPolynomial",
    "GaussianRationalPolynomialTerm",
    "UnitCircleCriticalPoint",
    "UnitCircleSupNormSquaredRequest",
    "UnitCircleSupNormSquaredResult",
]
