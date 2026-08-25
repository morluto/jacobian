"""Typed wire contracts for exact real algebra operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel

MAX_POLYNOMIAL_DEGREE = 32
MAX_POLYNOMIAL_TERMS = 33

# SymPy's ``sturm`` builds the exact Euclidean remainder sequence over QQ. For
# an integer-coefficient polynomial of degree ``n`` with ``d``-digit
# coefficients the chain coefficients grow like ``d * n^2`` decimal digits
# (verified adversarially), so a degree-32 input with 16-digit coefficients
# produces Sturm-chain coefficients of at most about 16,000 digits, comfortably
# inside the canonical 32,768-digit wire bound. Rational coefficients are
# rejected because a 16-digit numerator/denominator drives the plain QQ
# remainder sequence to roughly 200,000 digits, which cannot be represented.
MAX_COEFFICIENT_DIGITS = 16


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"polynomial.real_algebra_{reason}", message)


class PolynomialTerm(StrictModel):
    """One term: coefficient times x^exponent."""

    coefficient: CanonicalRational
    exponent: int = Field(ge=0, le=MAX_POLYNOMIAL_DEGREE)


class UnivariatePolynomial(StrictModel):
    """A univariate polynomial over QQ as sparse nonzero terms."""

    terms: tuple[PolynomialTerm, ...] = Field(
        min_length=1, max_length=MAX_POLYNOMIAL_TERMS
    )

    @model_validator(mode="after")
    def require_unique_exponents(self) -> Self:
        exponents = [t.exponent for t in self.terms]
        if len(set(exponents)) != len(exponents):
            raise _validation_error(
                "duplicate_exponents", "polynomial exponents must be unique"
            )
        if any(t.coefficient.as_fraction() == 0 for t in self.terms):
            raise _validation_error(
                "zero_term", "zero polynomial terms must be omitted"
            )
        return self


def _require_bounded_integer_coefficients(polynomial: UnivariatePolynomial) -> None:
    """Reject non-integer or oversized coefficients before Sturm construction."""

    for term in polynomial.terms:
        if term.coefficient.den != "1":
            raise _validation_error(
                "coefficient_domain", "polynomial coefficients must be integers"
            )
        require_bounded_rational(
            term.coefficient,
            max_digits=MAX_COEFFICIENT_DIGITS,
            label="polynomial coefficient",
        )


class SturmChainRequest(StrictModel):
    """Compute the Sturm chain of a univariate polynomial."""

    polynomial: UnivariatePolynomial

    @model_validator(mode="after")
    def require_nonconstant_polynomial(self) -> Self:
        if max(t.exponent for t in self.polynomial.terms) < 1:
            raise _validation_error(
                "constant_input", "Sturm chain requires a non-constant polynomial"
            )
        _require_bounded_integer_coefficients(self.polynomial)
        return self


class RootCountRequest(StrictModel):
    """Count real roots of a polynomial in an interval [lower, upper]."""

    polynomial: UnivariatePolynomial
    lower: CanonicalRational
    upper: CanonicalRational

    @model_validator(mode="after")
    def require_ordered_bounded_interval(self) -> Self:
        if self.lower.as_fraction() > self.upper.as_fraction():
            raise _validation_error(
                "interval_order", "lower bound must not exceed upper bound"
            )
        _require_bounded_integer_coefficients(self.polynomial)
        return self


class SturmChainResult(StrictModel):
    """The Sturm chain as a list of polynomials."""

    chain: tuple[UnivariatePolynomial, ...] = Field(min_length=1)
    degree: int = Field(ge=1, le=MAX_POLYNOMIAL_DEGREE)
    method: Literal["SYMPY_STURM"] = "SYMPY_STURM"


class RootCountResult(StrictModel):
    """Count of real roots in an interval."""

    root_count: int = Field(ge=0)
    lower: CanonicalRational
    upper: CanonicalRational
    method: Literal["STURM_THEOREM"] = "STURM_THEOREM"


__all__ = [
    "PolynomialTerm",
    "RootCountRequest",
    "RootCountResult",
    "SturmChainRequest",
    "SturmChainResult",
    "UnivariatePolynomial",
]
