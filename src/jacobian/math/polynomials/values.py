"""Provider-independent exact sparse rational-polynomial values."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel

PolynomialVariable = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,31}$", strict=True),
]
MAX_POLYNOMIAL_VARIABLES = 8
MAX_POLYNOMIAL_TERMS = 4_096
MAX_POLYNOMIAL_EXPONENT = 32_768


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"polynomial.{reason}", message)


class RationalPolynomialTerm(StrictModel):
    coefficient: CanonicalRational
    exponents: tuple[int, ...] = Field(
        min_length=0, max_length=MAX_POLYNOMIAL_VARIABLES
    )

    @model_validator(mode="after")
    def require_nonzero_coefficient_and_bounded_exponents(self) -> Self:
        if self.coefficient.as_fraction() == 0:
            raise _validation_error(
                "zero_term", "zero polynomial terms must be omitted"
            )
        if any(
            exponent < 0 or exponent > MAX_POLYNOMIAL_EXPONENT
            for exponent in self.exponents
        ):
            raise _validation_error(
                "exponent_bound",
                "polynomial exponents exceed the shared representation limit",
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
            raise _validation_error(
                "duplicate_exponents", "polynomial exponent tuples must be unique"
            )
        if exponents != tuple(sorted(exponents, reverse=True)):
            raise _validation_error(
                "term_order", "polynomial terms must use descending lexicographic order"
            )
        return self


class RationalPolynomial(StrictModel):
    """One sparse polynomial together with its exact coefficient ring."""

    domain: Literal["QQ"] = "QQ"
    variables: tuple[PolynomialVariable, ...] = Field(
        min_length=1, max_length=MAX_POLYNOMIAL_VARIABLES
    )
    polynomial: SparseRationalPolynomial

    @model_validator(mode="after")
    def require_matching_ring(self) -> Self:
        if len(set(self.variables)) != len(self.variables):
            raise _validation_error(
                "duplicate_variables", "polynomial variables must be unique"
            )
        if any(
            len(term.exponents) != len(self.variables) for term in self.polynomial.terms
        ):
            raise _validation_error(
                "monomial_shape",
                "every monomial must match the declared variable order",
            )
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
            raise _validation_error("ideal_variables", "ideal variables must be unique")
        if any(generator.variables != self.variables for generator in self.generators):
            raise _validation_error(
                "ideal_ring", "every ideal generator must use the declared ordered ring"
            )
        return self


class RationalFunction(StrictModel):
    """One reduced element of ``QQ(t_1, ..., t_n)``.

    The denominator is monic and the numerator and denominator are coprime.
    This makes the sparse representation unique for the declared variable
    order.  With no variables, the value is a canonical rational represented
    by one constant numerator over the constant denominator one.
    """

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
        raise _validation_error(
            "rational_function_variables", "rational-function variables must be unique"
        )
    for label, polynomial in (
        ("numerator", value.numerator),
        ("denominator", value.denominator),
    ):
        if any(
            len(term.exponents) != len(value.variables) for term in polynomial.terms
        ):
            raise _validation_error(
                "rational_function_shape",
                f"every {label} monomial must match the declared variable order",
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
        raise _validation_error(
            "zero_denominator", "rational-function denominator cannot be zero"
        )
    one = _rational_function_one(len(value.variables))
    if not value.numerator.terms:
        if value.denominator != one:
            raise _validation_error(
                "zero_normal_form", "canonical zero must have denominator one"
            )
        return
    if not value.variables:
        if value.denominator != one or len(value.numerator.terms) != 1:
            raise _validation_error(
                "constant_normal_form",
                "a rational function without variables must be a canonical rational",
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
        raise _validation_error(
            "denominator_not_monic", "rational-function denominator must be monic"
        )
    if not numerator.gcd(denominator).is_one:
        raise _validation_error(
            "not_coprime", "rational-function numerator and denominator must be coprime"
        )


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


def rational_evaluation_component_digit_bounds(
    polynomial: RationalPolynomial,
    point: tuple[CanonicalRational, ...],
) -> tuple[int, int]:
    """Bound numerator and denominator digits for exact point evaluation.

    The bound uses a common denominator formed from coefficient denominators
    and the greatest exponent of each point denominator. It is intentionally
    source-derived so evaluation can be rejected before exponentiation.
    """

    if len(point) != len(polynomial.variables):
        raise ValueError("evaluation point must match the polynomial axis")
    active_terms = tuple(
        term
        for term in polynomial.polynomial.terms
        if not any(
            exponent and coordinate.num == "0"
            for coordinate, exponent in zip(point, term.exponents, strict=True)
        )
    )
    if not active_terms:
        return 1, 1

    maximum_exponents = tuple(
        max(term.exponents[axis] for term in active_terms) for axis in range(len(point))
    )
    has_nontrivial_denominator = any(
        term.coefficient.den != "1" for term in active_terms
    ) or any(
        exponent and coordinate.den != "1"
        for coordinate, exponent in zip(point, maximum_exponents, strict=True)
    )
    common_denominator_digits = max(
        1,
        sum(
            len(term.coefficient.den)
            for term in active_terms
            if term.coefficient.den != "1"
        )
        + sum(
            exponent * len(coordinate.den)
            for coordinate, exponent in zip(point, maximum_exponents, strict=True)
            if coordinate.den != "1"
        ),
    )
    maximum_term_numerator_digits = max(
        max(
            1,
            (
                0
                if term.coefficient.num in {"1", "-1"}
                else len(term.coefficient.num.lstrip("-"))
            )
            + sum(
                exponent * len(coordinate.num.lstrip("-"))
                for coordinate, exponent in zip(point, term.exponents, strict=True)
                if coordinate.num not in {"1", "-1"}
            ),
        )
        for term in active_terms
    )
    denominator_scale_digits = (
        common_denominator_digits if has_nontrivial_denominator else 0
    )
    addition_digits = 0 if len(active_terms) == 1 else len(str(len(active_terms) - 1))
    numerator_digits = (
        maximum_term_numerator_digits + denominator_scale_digits + addition_digits
    )
    return numerator_digits, common_denominator_digits


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
    "rational_evaluation_component_digit_bounds",
    "require_polynomial_budget",
    "require_sparse_polynomial_budget",
]
