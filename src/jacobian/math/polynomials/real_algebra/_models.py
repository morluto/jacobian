"""Typed wire contracts for exact real algebra operations."""

from __future__ import annotations

from pydantic import Field
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer
from jacobian.math.polynomials.values import RationalPolynomial

MAX_POLYNOMIAL_DEGREE = 32
MAX_POLYNOMIAL_TERMS = 33

# Degree 32 and 16-digit primitive integer coefficients retain the established
# Euclidean-chain envelope. Rational inputs are admitted only when denominator
# clearing and content reduction fit that same envelope. SymPy sturm begins
# with the monic square-free part over QQ, so nonzero scalar rescaling gives
# the identical remainder sequence. Input scalar components remain capped.
MAX_COEFFICIENT_DIGITS = 16


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"polynomial.real_algebra_{reason}", message)


def _require_bounded_sturm_coefficients(polynomial: RationalPolynomial) -> None:
    """Bound rational input and its primitive integer representative before Sturm."""

    from math import gcd, lcm

    if (
        len(polynomial.variables) != 1
        or len(polynomial.polynomial.terms) > MAX_POLYNOMIAL_TERMS
        or any(
            term.exponents[0] > MAX_POLYNOMIAL_DEGREE
            for term in polynomial.polynomial.terms
        )
    ):
        raise _validation_error(
            "degree_bound", "Sturm input must be univariate of degree at most 32"
        )
    if not polynomial.polynomial.terms:
        raise _validation_error(
            "zero_polynomial",
            "the zero polynomial has no finite root count or Sturm chain",
        )
    for term in polynomial.polynomial.terms:
        require_bounded_rational(
            term.coefficient,
            max_digits=MAX_COEFFICIENT_DIGITS,
            label="polynomial coefficient",
        )

    denominators = [
        parse_canonical_integer(term.coefficient.den)
        for term in polynomial.polynomial.terms
    ]
    denominator = lcm(*denominators)
    coefficients = [
        parse_canonical_integer(term.coefficient.num) * (denominator // den)
        for term, den in zip(polynomial.polynomial.terms, denominators, strict=True)
    ]
    content = gcd(*coefficients)
    if any(
        abs(value // content) >= 10**MAX_COEFFICIENT_DIGITS for value in coefficients
    ):
        raise _validation_error(
            "coefficient_domain",
            "primitive integer coefficients exceed the Sturm height envelope",
        )


class SturmChainRequest(StrictModel):
    """Compute an ordinary Euclidean Sturm sequence for a bounded rational polynomial."""

    polynomial: RationalPolynomial


class RootCountRequest(StrictModel):
    """Count roots of a bounded rational polynomial in [lower, upper]."""

    polynomial: RationalPolynomial
    lower: CanonicalRational
    upper: CanonicalRational


class SturmChainResult(StrictModel):
    """The ordinary exact Euclidean Sturm sequence as polynomials."""

    chain: tuple[RationalPolynomial, ...] = Field(min_length=1)
    degree: int = Field(ge=1, le=MAX_POLYNOMIAL_DEGREE)


class RootCountResult(StrictModel):
    """A source-bound count of distinct real roots in a closed interval."""

    source_polynomial: RationalPolynomial
    root_count: int = Field(ge=0)
    lower: CanonicalRational
    upper: CanonicalRational


__all__ = [
    "RationalPolynomial",
    "RootCountRequest",
    "RootCountResult",
    "SturmChainRequest",
    "SturmChainResult",
]
