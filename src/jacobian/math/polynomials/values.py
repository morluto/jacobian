"""Provider-independent exact sparse rational-polynomial values."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel

PolynomialVariable = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,31}$", strict=True),
]
MAX_POLYNOMIAL_VARIABLES = 8
MAX_POLYNOMIAL_TERMS = 4_096
MAX_POLYNOMIAL_EXPONENT = 32_768


class RationalPolynomialTerm(StrictModel):
    coefficient: CanonicalRational
    exponents: tuple[int, ...] = Field(
        min_length=0, max_length=MAX_POLYNOMIAL_VARIABLES
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


class SparseRationalPolynomial(StrictModel):
    terms: tuple[RationalPolynomialTerm, ...] = Field(
        default=(),
        max_length=MAX_POLYNOMIAL_TERMS,
        description=(
            "Nonzero monomials in descending lexicographic order of their "
            "exponent tuples (highest first). For one variable, list [2] "
            "before [0]."
        ),
        examples=[
            [
                {
                    "coefficient": {"num": "1", "den": "1"},
                    "exponents": [2],
                },
                {
                    "coefficient": {"num": "-1", "den": "1"},
                    "exponents": [0],
                },
            ]
        ],
    )

    @model_validator(mode="after")
    def require_unique_canonical_term_order(self) -> Self:
        exponents = tuple(term.exponents for term in self.terms)
        if len(set(exponents)) != len(exponents):
            raise ValueError("polynomial exponent tuples must be unique")
        if exponents != tuple(sorted(exponents, reverse=True)):
            raise ValueError("polynomial terms must use descending lexicographic order")
        return self


class RationalPolynomial(StrictModel):
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


class RationalPolynomialIdeal(StrictModel):
    """A finitely generated ideal in one explicitly ordered ``QQ`` ring.

    Generator lists are presentations, not canonical bases.  Every generator
    nevertheless carries the same authoritative ring so ideals can pass
    directly between operations without rendering or reparsing expressions.
    """

    variables: tuple[PolynomialVariable, ...] = Field(
        min_length=1,
        max_length=MAX_POLYNOMIAL_VARIABLES,
    )
    generators: tuple[RationalPolynomial, ...] = Field(
        min_length=1,
        max_length=64,
    )

    @model_validator(mode="after")
    def require_one_ordered_ring(self) -> Self:
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("ideal variables must be unique")
        if any(generator.variables != self.variables for generator in self.generators):
            raise ValueError("every ideal generator must use the declared ordered ring")
        return self


class RationalFunction(StrictModel):
    """One reduced element of ``QQ(t_1, ..., t_n)``.

    The denominator is monic and the numerator and denominator are coprime.
    This makes the sparse representation unique for the declared variable
    order.  With no variables, the value is a canonical rational represented
    by one constant numerator over the constant denominator one.
    """

    rational_function_schema_version: Literal["1"] = "1"
    domain: Literal["QQ"] = "QQ"
    variables: tuple[PolynomialVariable, ...] = Field(
        min_length=0,
        max_length=MAX_POLYNOMIAL_VARIABLES,
    )
    numerator: SparseRationalPolynomial
    denominator: SparseRationalPolynomial

    @model_validator(mode="after")
    def require_canonical_fraction(self) -> Self:
        _require_rational_function_shapes(self)
        _require_rational_function_normal_form(self)
        return self


def _rational_function_one(variable_count: int) -> SparseRationalPolynomial:
    return SparseRationalPolynomial(
        terms=(
            RationalPolynomialTerm(
                coefficient=CanonicalRational(num="1", den="1"),
                exponents=(0,) * variable_count,
            ),
        )
    )


def _require_rational_function_shapes(value: RationalFunction) -> None:
    if len(set(value.variables)) != len(value.variables):
        raise ValueError("rational-function variables must be unique")
    for label, polynomial in (
        ("numerator", value.numerator),
        ("denominator", value.denominator),
    ):
        if any(
            len(term.exponents) != len(value.variables) for term in polynomial.terms
        ):
            raise ValueError(
                f"every {label} monomial must match the declared variable order"
            )
        require_sparse_polynomial_budget(
            polynomial,
            maximum_terms=256,
            maximum_exponent=64,
            maximum_coefficient_digits=128,
            label=f"rational-function {label}",
        )


def _require_rational_function_normal_form(value: RationalFunction) -> None:
    if not value.denominator.terms:
        raise ValueError("rational-function denominator cannot be zero")
    one = _rational_function_one(len(value.variables))
    if not value.numerator.terms:
        if value.denominator != one:
            raise ValueError("canonical zero must have denominator one")
        return
    if not value.variables:
        if value.denominator != one or len(value.numerator.terms) != 1:
            raise ValueError(
                "a rational function without variables must be a canonical rational"
            )
        return

    # Construct exact polynomials from already validated term data. No caller
    # text is parsed or evaluated at this boundary.
    from jacobian.math.polynomials._conversions import (
        sparse_rational_polynomial_to_sympy,
    )

    numerator = sparse_rational_polynomial_to_sympy(value.numerator, value.variables)
    denominator = sparse_rational_polynomial_to_sympy(
        value.denominator, value.variables
    )
    if denominator.LC() != 1:
        raise ValueError("rational-function denominator must be monic")
    if not numerator.gcd(denominator).is_one:
        raise ValueError("rational-function numerator and denominator must be coprime")


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
    "RationalFunction",
    "RationalPolynomial",
    "RationalPolynomialIdeal",
    "RationalPolynomialTerm",
    "SparseRationalPolynomial",
    "require_polynomial_budget",
    "require_sparse_polynomial_budget",
]
