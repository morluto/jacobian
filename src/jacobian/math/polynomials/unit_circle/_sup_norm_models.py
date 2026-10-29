"""Typed contracts for exact unit-circle supremum norms of polynomials.

The main result is a certified rational enclosure together with the complete
critical-point ledger.  The maximum of ``|P(z)|^2`` on ``|z| = 1`` is the
value of the rational function ``Q(t) / (1 + t**2)**degree`` at a real root of
an exact derivative numerator.  Its minimal polynomial divides the critical
resultant, whose ``y``-degree is the derivative-numerator degree; the two
leading ``t**(2m+1)`` terms of the derivative numerator cancel exactly, so
that degree is at most twice the input degree (hence at most 16 here) and the
minimal factor always fits the shared degree-16 real-algebraic carrier.  The
exact value is therefore always bound as an indexed root except in the
theoretical coefficient-height overflow case, where the field is ``None`` and
the enclosure remains the complete certificate.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictBool, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.number_theory.algebraic_numbers.real import (
    RationalIsolatingInterval,
    RealAlgebraicValue,
)
from jacobian.math.number_theory.number_fields import GaussianRational

# The transformed numerator has degree at most ``2 * degree`` and the exact
# derivative numerator at most ``2 * degree + 1``.  Exact real-root isolation
# is performed on that derivative numerator, so the number of real critical
# points is bounded by its degree.  The degree limit keeps the pairwise exact
# comparison ledger and the resultant-based value identification inside the
# degree/height envelope of maintained FLINT elimination/factorization and
# SymPy real-root isolation.
MAX_SUP_NORM_DEGREE = 8
MAX_SUP_NORM_TERMS = MAX_SUP_NORM_DEGREE + 1
MAX_SUP_NORM_INPUT_COMPONENT_DIGITS = 32
MAX_SUP_NORM_NUMERATOR_DEGREE = 2 * MAX_SUP_NORM_DEGREE
MAX_SUP_NORM_DERIVATIVE_DEGREE = 2 * MAX_SUP_NORM_DEGREE + 1
MAX_SUP_NORM_CRITICAL_ROOTS = MAX_SUP_NORM_DERIVATIVE_DEGREE
# Raw Cayley expansion uses a common denominator squared. Its derivative
# cancels every diagonal scalar square and needs each input denominator only
# once. The denominator-aware admission proves these separate height bounds.
MAX_SUP_NORM_TRANSFORM_COMPONENT_DIGITS = 1_216
MAX_SUP_NORM_DERIVATIVE_COMPONENT_DIGITS = 640
MAX_SUP_NORM_RESULTANT_COMPONENT_DIGITS = 32_768
MAX_SUP_NORM_PARAMETER_COMPONENT_DIGITS = MAX_CANONICAL_RATIONAL_DIGITS
# A 10000-place outward grid is finer than the derivative root separation.
MAX_SUP_NORM_PARAMETER_DECIMAL_PLACES = 10_000
# One comparison ledger entry per distinct critical value plus the z=-1
# endpoint; this is the complete pairwise comparison budget.
MAX_SUP_NORM_COMPARISONS = (
    (MAX_SUP_NORM_CRITICAL_ROOTS + 1) * (MAX_SUP_NORM_CRITICAL_ROOTS + 2)
) // 2
# Nonsingleton reported value/image intervals use an outward decimal grid;
# exact rational singletons retain their separately bounded source heights.
MAX_SUP_NORM_ENCLOSURE_COMPONENT_DIGITS = 256
MAX_SUP_NORM_CERTIFICATE_SCALAR_DIGITS = 1 << 22
MAX_SUP_NORM_TRANSFORM_WORK = 8_000_000
MAX_SUP_NORM_ISOLATION_WORK = 160_000_000
MAX_SUP_NORM_RESULTANT_WORK = 1 << 46
MAX_SUP_NORM_FACTOR_WORK = 1 << 58
MAX_SUP_NORM_VALUE_ANALYSIS_WORK = 1 << 58
MAX_SUP_NORM_POSTPROCESSING_WORK = 1 << 55


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"polynomial.unit_circle.sup_norm_{reason}", message)


class GaussianRationalPolynomialTerm(StrictModel):
    """One nonzero Gaussian-rational monomial of the input polynomial."""

    coefficient: GaussianRational
    exponent: StrictInt = Field(ge=0, le=MAX_SUP_NORM_DEGREE)

    @model_validator(mode="after")
    def require_nonzero_coefficient(self) -> Self:
        if (
            self.coefficient.real.as_fraction() == 0
            and self.coefficient.imaginary.as_fraction() == 0
        ):
            raise _validation_error(
                "zero_term", "zero polynomial terms must be omitted"
            )
        return self


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

    polynomial: GaussianRationalPolynomial = Field(
        description=(
            "Polynomial of degree at most 8 with numerator/denominator "
            "components of at most 32 decimal digits. Admission separately "
            "bounds denominator-aware Cayley expansion, real-root isolation, "
            "resultants, factors, and retained exact scalars."
        )
    )


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
    ``sup_norm_squared_exact`` carries the maximum itself as an indexed root
    of its irreducible resultant factor.  The factor degree is at most twice
    the input degree (leading-term cancellation in the derivative numerator),
    hence always inside the shared carrier; the field is ``None`` only in the
    theoretical coefficient-height overflow case, in which the enclosure is
    the complete certificate.
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
    sup_norm_squared_exact: RealAlgebraicValue | None = None
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
        sup_norm_squared_exact: RealAlgebraicValue | None,
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
            sup_norm_squared_exact=sup_norm_squared_exact,
            sup_norm_enclosure=sup_norm_enclosure,
        )


__all__ = [
    "MAX_SUP_NORM_CERTIFICATE_SCALAR_DIGITS",
    "MAX_SUP_NORM_COMPARISONS",
    "MAX_SUP_NORM_CRITICAL_ROOTS",
    "MAX_SUP_NORM_DEGREE",
    "MAX_SUP_NORM_DERIVATIVE_COMPONENT_DIGITS",
    "MAX_SUP_NORM_DERIVATIVE_DEGREE",
    "MAX_SUP_NORM_ENCLOSURE_COMPONENT_DIGITS",
    "MAX_SUP_NORM_FACTOR_WORK",
    "MAX_SUP_NORM_INPUT_COMPONENT_DIGITS",
    "MAX_SUP_NORM_ISOLATION_WORK",
    "MAX_SUP_NORM_NUMERATOR_DEGREE",
    "MAX_SUP_NORM_PARAMETER_COMPONENT_DIGITS",
    "MAX_SUP_NORM_PARAMETER_DECIMAL_PLACES",
    "MAX_SUP_NORM_POSTPROCESSING_WORK",
    "MAX_SUP_NORM_RESULTANT_COMPONENT_DIGITS",
    "MAX_SUP_NORM_RESULTANT_WORK",
    "MAX_SUP_NORM_TERMS",
    "MAX_SUP_NORM_TRANSFORM_COMPONENT_DIGITS",
    "MAX_SUP_NORM_TRANSFORM_WORK",
    "MAX_SUP_NORM_VALUE_ANALYSIS_WORK",
    "GaussianRationalPolynomial",
    "GaussianRationalPolynomialTerm",
    "UnitCircleCriticalPoint",
    "UnitCircleSupNormSquaredRequest",
    "UnitCircleSupNormSquaredResult",
]
