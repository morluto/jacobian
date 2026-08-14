"""Provider-independent exact sparse rational-polynomial values."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from jacobian.contracts.base import ContractModel
from jacobian.contracts.exact import CanonicalRational, require_bounded_rational

PolynomialVariable = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,31}$", strict=True),
]
MAX_POLYNOMIAL_VARIABLES = 8
MAX_POLYNOMIAL_TERMS = 4_096
MAX_POLYNOMIAL_EXPONENT = 32_768


class RationalPolynomialTerm(ContractModel):
    coefficient: CanonicalRational
    exponents: tuple[int, ...] = Field(
        min_length=1, max_length=MAX_POLYNOMIAL_VARIABLES
    )

    @model_validator(mode="after")
    def require_nonzero_coefficient_and_bounded_exponents(self) -> Self:
        if self.coefficient.as_fraction() == 0:
            raise ValueError("zero polynomial terms must be omitted")
        if any(
            exponent < 0 or exponent > MAX_POLYNOMIAL_EXPONENT
            for exponent in self.exponents
        ):
            raise ValueError(
                "polynomial exponents exceed the shared representation limit"
            )
        return self


class SparseRationalPolynomial(ContractModel):
    terms: tuple[RationalPolynomialTerm, ...] = Field(
        default=(), max_length=MAX_POLYNOMIAL_TERMS
    )

    @model_validator(mode="after")
    def require_unique_canonical_term_order(self) -> Self:
        exponents = tuple(term.exponents for term in self.terms)
        if len(set(exponents)) != len(exponents):
            raise ValueError("polynomial exponent tuples must be unique")
        if exponents != tuple(sorted(exponents, reverse=True)):
            raise ValueError("polynomial terms must use descending lexicographic order")
        return self


class RationalPolynomial(ContractModel):
    """One sparse polynomial together with its exact coefficient ring."""

    polynomial_schema_version: Literal["1"] = "1"
    domain: Literal["QQ"] = "QQ"
    variables: tuple[PolynomialVariable, ...] = Field(
        min_length=1, max_length=MAX_POLYNOMIAL_VARIABLES
    )
    polynomial: SparseRationalPolynomial

    @model_validator(mode="after")
    def require_matching_ring(self) -> Self:
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("polynomial variables must be unique")
        if any(
            len(term.exponents) != len(self.variables) for term in self.polynomial.terms
        ):
            raise ValueError("every monomial must match the declared variable order")
        return self


def require_sparse_polynomial_budget(
    polynomial: SparseRationalPolynomial,
    *,
    maximum_terms: int,
    maximum_exponent: int,
    maximum_coefficient_digits: int = 256,
    label: str = "polynomial",
) -> None:
    """Apply an operation-owned cost budget to one polynomial value."""

    if len(polynomial.terms) > maximum_terms:
        raise ValueError(f"{label} exceeds the {maximum_terms}-term operation budget")
    for term in polynomial.terms:
        require_bounded_rational(
            term.coefficient,
            max_digits=maximum_coefficient_digits,
            label=f"{label} coefficient",
        )
        if any(exponent > maximum_exponent for exponent in term.exponents):
            raise ValueError(
                f"{label} exponent exceeds the {maximum_exponent}-degree operation budget"
            )


def require_polynomial_budget(
    polynomial: RationalPolynomial,
    *,
    maximum_terms: int,
    maximum_exponent: int,
    maximum_coefficient_digits: int = 256,
    label: str = "polynomial",
) -> None:
    """Apply an operation-owned cost budget to an authoritative polynomial."""

    require_sparse_polynomial_budget(
        polynomial.polynomial,
        maximum_terms=maximum_terms,
        maximum_exponent=maximum_exponent,
        maximum_coefficient_digits=maximum_coefficient_digits,
        label=label,
    )


__all__ = [
    "MAX_POLYNOMIAL_EXPONENT",
    "MAX_POLYNOMIAL_TERMS",
    "MAX_POLYNOMIAL_VARIABLES",
    "PolynomialVariable",
    "RationalPolynomial",
    "RationalPolynomialTerm",
    "SparseRationalPolynomial",
    "require_polynomial_budget",
    "require_sparse_polynomial_budget",
]
