"""Contracts and bounds for exact Gaussian polynomial moments."""

from __future__ import annotations

from fractions import Fraction
from itertools import pairwise
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import (
    CanonicalInteger,
    CanonicalRational,
    require_bounded_rational,
)
from jacobian._models import StrictModel
from jacobian.math.probability._gaussian_moments import gaussian_univariate_moment

MAX_GAUSSIAN_VARIABLES = 16
MAX_GAUSSIAN_POLYNOMIAL_TERMS = 16
MAX_GAUSSIAN_TERM_DEGREE = 8
MAX_GAUSSIAN_MOMENT_ORDER = 16
MAX_GAUSSIAN_EXPANSION_PATHS = 65536
MAX_GAUSSIAN_RESULT_RATIONAL_DIGITS = 4096


def _validation_error(message: str) -> PydanticCustomError:
    return PydanticCustomError("probability.model_invariant", message)


class ExactComplexRational(StrictModel):
    """One exact element of Q(i), encoded without floating-point values."""

    real: CanonicalRational
    imaginary: CanonicalRational

    def as_fractions(self) -> tuple[Fraction, Fraction]:
        return self.real.as_fraction(), self.imaginary.as_fraction()

    @model_validator(mode="after")
    def require_bounded_components(self) -> Self:
        for label, value in (
            ("complex real component", self.real),
            ("complex imaginary component", self.imaginary),
        ):
            require_bounded_rational(
                value,
                max_digits=MAX_GAUSSIAN_RESULT_RATIONAL_DIGITS,
                label=label,
            )
        return self


class GaussianPolynomialTerm(StrictModel):
    coefficient: ExactComplexRational
    exponents: tuple[StrictInt, ...] = Field(
        min_length=1,
        max_length=MAX_GAUSSIAN_VARIABLES,
    )

    @model_validator(mode="after")
    def require_bounded_nonzero_term(self) -> Self:
        if any(
            type(exponent) is not int or exponent < 0 for exponent in self.exponents
        ):
            raise _validation_error(
                "Gaussian polynomial exponents must be nonnegative integers"
            )
        if sum(self.exponents) > MAX_GAUSSIAN_TERM_DEGREE:
            raise _validation_error(
                "Gaussian polynomial term exceeds the "
                f"{MAX_GAUSSIAN_TERM_DEGREE}-degree bound"
            )
        if self.coefficient.as_fractions() == (Fraction(), Fraction()):
            raise _validation_error(
                "Gaussian polynomial terms must have nonzero coefficients"
            )
        return self


class GaussianPolynomial(StrictModel):
    """A canonical sparse polynomial in independent standard real Gaussians."""

    variable_count: StrictInt = Field(ge=1, le=MAX_GAUSSIAN_VARIABLES)
    terms: tuple[GaussianPolynomialTerm, ...] = Field(
        min_length=1,
        max_length=MAX_GAUSSIAN_POLYNOMIAL_TERMS,
        description=(
            "Nonzero sparse terms ordered lexicographically by their complete "
            "exponent vectors, for example [0, 1] before [1, 0]."
        ),
    )

    @model_validator(mode="after")
    def require_canonical_sparse_polynomial(self) -> Self:
        exponents = tuple(term.exponents for term in self.terms)
        if any(len(item) != self.variable_count for item in exponents):
            raise _validation_error(
                "every Gaussian polynomial exponent vector must match variable_count"
            )
        for left, right in pairwise(exponents):
            if left >= right:
                raise _validation_error(
                    "Gaussian polynomial terms must use strictly increasing "
                    "lexicographic exponent-vector order; first offending adjacent "
                    f"pair is {list(left)} then {list(right)}"
                )
        return self


class GaussianPolynomialMomentRequest(StrictModel):
    polynomial: GaussianPolynomial
    order: StrictInt = Field(ge=0, le=MAX_GAUSSIAN_MOMENT_ORDER)

    @model_validator(mode="after")
    def require_bounded_complete_expansion(self) -> Self:
        expansion_paths = len(self.polynomial.terms) ** self.order
        if expansion_paths > MAX_GAUSSIAN_EXPANSION_PATHS:
            raise _validation_error(
                "Gaussian polynomial power exceeds the "
                f"{MAX_GAUSSIAN_EXPANSION_PATHS}-path expansion bound"
            )
        components = tuple(
            component
            for term in self.polynomial.terms
            for component in (term.coefficient.real, term.coefficient.imaginary)
        )
        distinct_denominator_digits = sum(
            len(denominator)
            for denominator in {component.den for component in components}
        )
        maximum_numerator_digits = max(
            len(component.num.lstrip("-")) for component in components
        )
        result_digit_bound = (
            self.order * (distinct_denominator_digits + maximum_numerator_digits)
            + len(str(max(1, expansion_paths)))
            + 64
        )
        if result_digit_bound > MAX_GAUSSIAN_RESULT_RATIONAL_DIGITS:
            raise _validation_error(
                "Gaussian polynomial coefficient denominators can exceed the "
                f"{MAX_GAUSSIAN_RESULT_RATIONAL_DIGITS}-digit result bound"
            )
        return self


class GaussianMomentContraction(StrictModel):
    exponents: tuple[StrictInt, ...] = Field(
        min_length=1,
        max_length=MAX_GAUSSIAN_VARIABLES,
    )
    expanded_coefficient: ExactComplexRational
    variable_moment_factors: tuple[CanonicalInteger, ...] = Field(
        min_length=1,
        max_length=MAX_GAUSSIAN_VARIABLES,
    )
    gaussian_moment_factor: CanonicalInteger
    contribution: ExactComplexRational

    @model_validator(mode="after")
    def bind_gaussian_contraction(self) -> Self:
        if len(self.exponents) != len(self.variable_moment_factors):
            raise _validation_error("Gaussian contraction dimensions disagree")
        expected_factors = tuple(
            gaussian_univariate_moment(exponent) for exponent in self.exponents
        )
        actual_factors = tuple(int(value) for value in self.variable_moment_factors)
        if actual_factors != expected_factors:
            raise _validation_error("Gaussian variable moment factors are invalid")
        expected_factor = 1
        for factor in expected_factors:
            expected_factor *= factor
        if int(self.gaussian_moment_factor) != expected_factor:
            raise _validation_error(
                "Gaussian moment factor does not match its variables"
            )
        coefficient = self.expanded_coefficient.as_fractions()
        contribution = self.contribution.as_fractions()
        if contribution != (
            coefficient[0] * expected_factor,
            coefficient[1] * expected_factor,
        ):
            raise _validation_error("Gaussian contraction contribution is invalid")
        return self


class GaussianPolynomialMomentResult(StrictModel):
    order: StrictInt = Field(ge=0, le=MAX_GAUSSIAN_MOMENT_ORDER)
    moment: ExactComplexRational
    expansion_path_count: StrictInt = Field(ge=1, le=MAX_GAUSSIAN_EXPANSION_PATHS)
    expanded_monomial_count: StrictInt = Field(ge=1, le=MAX_GAUSSIAN_EXPANSION_PATHS)
    contractions: tuple[GaussianMomentContraction, ...] = Field(
        min_length=1,
        max_length=MAX_GAUSSIAN_EXPANSION_PATHS,
    )
    gaussian_model: Literal["INDEPENDENT_STANDARD_REAL"] = "INDEPENDENT_STANDARD_REAL"
    completeness: Literal["COMPLETE_BOUNDED_EXPANSION"] = "COMPLETE_BOUNDED_EXPANSION"
    exactness: Literal["EXACT_COMPLEX_RATIONAL"] = "EXACT_COMPLEX_RATIONAL"
    determinism: Literal["DETERMINISTIC"] = "DETERMINISTIC"

    @model_validator(mode="after")
    def bind_complete_contraction_ledger(self) -> Self:
        if self.expanded_monomial_count != len(self.contractions):
            raise _validation_error("expanded monomial count does not match the ledger")
        exponents = tuple(item.exponents for item in self.contractions)
        if any(left >= right for left, right in pairwise(exponents)):
            raise _validation_error(
                "Gaussian contractions must use strictly increasing exponent order"
            )
        if any(len(item) != len(exponents[0]) for item in exponents):
            raise _validation_error("Gaussian contraction dimensions disagree")
        total_real = Fraction()
        total_imaginary = Fraction()
        for item in self.contractions:
            real, imaginary = item.contribution.as_fractions()
            total_real += real
            total_imaginary += imaginary
        if self.moment.as_fractions() != (total_real, total_imaginary):
            raise _validation_error(
                "Gaussian polynomial moment does not match its ledger"
            )
        return self


__all__ = [
    "MAX_GAUSSIAN_EXPANSION_PATHS",
    "MAX_GAUSSIAN_MOMENT_ORDER",
    "MAX_GAUSSIAN_POLYNOMIAL_TERMS",
    "MAX_GAUSSIAN_RESULT_RATIONAL_DIGITS",
    "MAX_GAUSSIAN_TERM_DEGREE",
    "MAX_GAUSSIAN_VARIABLES",
    "ExactComplexRational",
    "GaussianMomentContraction",
    "GaussianPolynomial",
    "GaussianPolynomialMomentRequest",
    "GaussianPolynomialMomentResult",
    "GaussianPolynomialTerm",
]
