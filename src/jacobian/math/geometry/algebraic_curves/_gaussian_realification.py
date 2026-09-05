"""Exact realification of univariate Gaussian-rational polynomials."""

from __future__ import annotations

from fractions import Fraction
from math import comb
from typing import Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.polynomials.values import (
    MAX_RATIONAL_FUNCTION_EXPONENT,
    MAX_RATIONAL_FUNCTION_TERMS,
    PolynomialVariable,
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)
from jacobian.math.probability import ExactComplexRational

MAX_GAUSSIAN_REALIFICATION_TERMS = 64
MAX_GAUSSIAN_REALIFICATION_DEGREE = 64
MAX_GAUSSIAN_REALIFICATION_COEFFICIENT_DIGITS = 256
MAX_GAUSSIAN_REALIFICATION_EXPANDED_TERMS = 4096
MAX_GAUSSIAN_REALIFICATION_RESULT_TERMS = 256


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"algebraic_geometry.{reason}", message)


# Q(i) has one shared serialized value across Jacobian.  Keep this alias for
# existing Python callers while making the field and wire representation compose
# with the Gaussian-moment operations.
GaussianComplexCoefficient = ExactComplexRational


class UnivariateGaussianPolynomialTerm(StrictModel):
    coefficient: GaussianComplexCoefficient
    exponent: StrictInt = Field(ge=0, le=MAX_GAUSSIAN_REALIFICATION_DEGREE)

    @model_validator(mode="after")
    def require_nonzero(self) -> Self:
        real, imag = self.coefficient.as_fractions()
        if real == 0 and imag == 0:
            raise _validation_error(
                "gaussian_term_zero_coefficient",
                "Gaussian polynomial terms must have nonzero coefficient",
            )
        return self


class UnivariateGaussianPolynomial(StrictModel):
    """Sparse univariate polynomial over Q(i) in one named variable."""

    variable: PolynomialVariable
    terms: tuple[UnivariateGaussianPolynomialTerm, ...] = Field(
        default=(),
        max_length=MAX_GAUSSIAN_REALIFICATION_TERMS,
        description=(
            "Sparse terms sorted strictly decreasing by exponent; empty tuple "
            "denotes the zero polynomial. Each exponent is distinct."
        ),
    )

    @model_validator(mode="after")
    def require_canonical(self) -> Self:
        exponents = tuple(term.exponent for term in self.terms)
        if len(set(exponents)) != len(exponents):
            raise _validation_error(
                "gaussian_polynomial_duplicate_exponent",
                "Gaussian polynomial exponents must be unique",
            )
        if exponents != tuple(sorted(exponents, reverse=True)):
            raise _validation_error(
                "gaussian_polynomial_term_order",
                "Gaussian polynomial terms must be strictly decreasing by exponent",
            )
        return self


class GaussianRealificationRequest(StrictModel):
    """Request the exact real and imaginary parts of p(x+i y)."""

    polynomial: UnivariateGaussianPolynomial
    target_variables: tuple[PolynomialVariable, PolynomialVariable] = Field(
        min_length=2,
        max_length=2,
        description="Ordered pair (x,y) for the realification, distinct from the source variable.",
    )

    @model_validator(mode="before")
    @classmethod
    def bound_raw(cls, value: object) -> object:
        return canonicalize_json_containers(value)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if len(set(self.target_variables)) != 2:
            raise _validation_error(
                "gaussian_target_variables_not_unique",
                "target variables must be distinct",
            )
        if self.polynomial.variable in self.target_variables:
            raise _validation_error(
                "gaussian_target_collides_with_source",
                "target variables must be distinct from the source variable",
            )
        return self


class GaussianRealificationResult(StrictModel):
    """Real and imaginary parts of p(x+i y) as bivariate rational polynomials."""

    source_polynomial: UnivariateGaussianPolynomial
    target_variables: tuple[PolynomialVariable, PolynomialVariable]
    real_part: RationalPolynomial
    imag_part: RationalPolynomial
    substitution: str = Field(
        description=(
            "Display-only, non-evaluated substitution with the exact grammar "
            "'<source> = <first-target> + i*<second-target>', derived from "
            "the retained variables."
        ),
        max_length=128,
    )

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if len(set(self.target_variables)) != 2:
            raise _validation_error(
                "gaussian_target_variables_not_unique",
                "target variables must be distinct",
            )
        if self.source_polynomial.variable in self.target_variables:
            raise _validation_error(
                "gaussian_target_collides_with_source",
                "target variables must be distinct from the source variable",
            )
        if self.real_part.variables != self.target_variables:
            raise _validation_error(
                "gaussian_real_part_variables",
                "real part must use the ordered target variables",
            )
        if self.imag_part.variables != self.target_variables:
            raise _validation_error(
                "gaussian_imag_part_variables",
                "imag part must use the ordered target variables",
            )
        expected = f"{self.source_polynomial.variable} = {self.target_variables[0]} + i*{self.target_variables[1]}"
        if self.substitution != expected:
            raise _validation_error(
                "gaussian_substitution_mismatch",
                f"substitution must be '{expected}'",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        source_polynomial: UnivariateGaussianPolynomial,
        target_variables: tuple[PolynomialVariable, PolynomialVariable],
        real_part: RationalPolynomial,
        imag_part: RationalPolynomial,
    ) -> Self:
        substitution = f"{source_polynomial.variable} = {target_variables[0]} + i*{target_variables[1]}"
        return cls.model_construct(
            source_polynomial=source_polynomial,
            target_variables=target_variables,
            real_part=real_part,
            imag_part=imag_part,
            substitution=substitution,
        )


def _admit_gaussian_realification(
    poly: UnivariateGaussianPolynomial,
    target_variables: tuple[PolynomialVariable, PolynomialVariable],
) -> None:
    if len(set(target_variables)) != 2:
        raise OperationDomainValidationError(
            location=("target_variables",),
            code="algebraic_geometry.gaussian_target_variables_not_unique",
            message="target variables must be distinct",
        )
    if poly.variable in target_variables:
        raise OperationDomainValidationError(
            location=("target_variables",),
            code="algebraic_geometry.gaussian_target_collides_with_source",
            message="target variables must be distinct from the source variable",
        )
    # Source bounds already enforced by model, but check predicted expansion
    degree = max((t.exponent for t in poly.terms), default=0)
    term_count = len(poly.terms)
    if term_count > MAX_GAUSSIAN_REALIFICATION_TERMS:
        raise OperationDomainValidationError(
            location=("polynomial", "terms"),
            code="algebraic_geometry.gaussian_realification.term_count",
            message=f"Gaussian polynomial exceeds {MAX_GAUSSIAN_REALIFICATION_TERMS} terms",
        )
    if degree > MAX_GAUSSIAN_REALIFICATION_DEGREE:
        raise OperationDomainValidationError(
            location=("polynomial", "terms"),
            code="algebraic_geometry.gaussian_realification.degree",
            message=f"Gaussian polynomial degree exceeds {MAX_GAUSSIAN_REALIFICATION_DEGREE}",
        )
    # Predicted raw expansion count: sum_{terms} (exponent+1)
    predicted_raw = sum(t.exponent + 1 for t in poly.terms)
    if predicted_raw > MAX_GAUSSIAN_REALIFICATION_EXPANDED_TERMS:
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="algebraic_geometry.gaussian_realification.expanded_terms",
            message=f"predicted expansion {predicted_raw} exceeds {MAX_GAUSSIAN_REALIFICATION_EXPANDED_TERMS}",
        )
    real_terms = sum(
        sum(
            1
            for j in range(term.exponent + 1)
            if (j % 2 == 0 and term.coefficient.real.as_fraction() != 0)
            or (j % 2 == 1 and term.coefficient.imaginary.as_fraction() != 0)
        )
        for term in poly.terms
    )
    imag_terms = sum(
        sum(
            1
            for j in range(term.exponent + 1)
            if (j % 2 == 1 and term.coefficient.real.as_fraction() != 0)
            or (j % 2 == 0 and term.coefficient.imaginary.as_fraction() != 0)
        )
        for term in poly.terms
    )
    if max(real_terms, imag_terms) > MAX_GAUSSIAN_REALIFICATION_RESULT_TERMS:
        raise OperationDomainValidationError(
            location=("polynomial", "terms"),
            code="algebraic_geometry.gaussian_realification.result_terms",
            message=(
                f"a realification component may contain {max(real_terms, imag_terms)} terms, "
                f"exceeding {MAX_GAUSSIAN_REALIFICATION_RESULT_TERMS}"
            ),
        )

    # A binomial multiplier can enlarge a numerator. Reserve that room during
    # admission so result construction never rejects an accepted coefficient.
    for index, term in enumerate(poly.terms):
        for label, value in (
            ("real", term.coefficient.real),
            ("imaginary", term.coefficient.imaginary),
        ):
            coefficient = Fraction(int(value.num), int(value.den))
            maximum_digits = max(
                max(
                    len(str(abs((coefficient * comb(term.exponent, j)).numerator))),
                    len(str((coefficient * comb(term.exponent, j)).denominator)),
                )
                for j in range(term.exponent + 1)
            )
            if maximum_digits > MAX_GAUSSIAN_REALIFICATION_COEFFICIENT_DIGITS:
                raise OperationDomainValidationError(
                    location=("polynomial", "terms", index, "coefficient", label),
                    code="algebraic_geometry.gaussian_realification.coefficient_digits",
                    message="Gaussian coefficient exceeds the result digit envelope after binomial multiplication",
                )


def _fraction_to_canonical_rational(value: Fraction) -> CanonicalRational:
    return CanonicalRational(
        num=format_canonical_integer(value.numerator),
        den=format_canonical_integer(value.denominator),
    )


def _build_rational_polynomial(
    variables: tuple[PolynomialVariable, PolynomialVariable],
    monomials: dict[tuple[int, int], Fraction],
) -> RationalPolynomial:
    # Remove zeros, sort descending lexicographic, check bounds
    filtered = {exp: coeff for exp, coeff in monomials.items() if coeff != 0}
    if not filtered:
        sparse = SparseRationalPolynomial(terms=())
        return RationalPolynomial(domain="QQ", variables=variables, polynomial=sparse)
    # Sort descending lexicographic: compare exponents tuple reverse=True
    sorted_exps = sorted(filtered.keys(), reverse=True)
    if len(sorted_exps) > MAX_RATIONAL_FUNCTION_TERMS:
        raise OperationDomainValidationError(
            location=("target",),
            code="algebraic_geometry.gaussian_realification.result_terms",
            message=f"realification result exceeds {MAX_RATIONAL_FUNCTION_TERMS} terms",
        )
    for exp in sorted_exps:
        if any(e < 0 or e > MAX_RATIONAL_FUNCTION_EXPONENT for e in exp):
            raise OperationDomainValidationError(
                location=("target",),
                code="algebraic_geometry.gaussian_realification.exponent",
                message=f"exponent {exp} exceeds {MAX_RATIONAL_FUNCTION_EXPONENT}",
            )
    terms = tuple(
        RationalPolynomialTerm(
            coefficient=_fraction_to_canonical_rational(filtered[exp]),
            exponents=exp,
        )
        for exp in sorted_exps
    )
    # Validate coefficient digits
    for term in terms:
        require_bounded_rational(
            term.coefficient,
            max_digits=MAX_GAUSSIAN_REALIFICATION_COEFFICIENT_DIGITS,
            label="realification result coefficient",
        )
    sparse = SparseRationalPolynomial(terms=terms)
    return RationalPolynomial(domain="QQ", variables=variables, polynomial=sparse)


def gaussian_realification(
    polynomial: UnivariateGaussianPolynomial,
    target_variables: tuple[PolynomialVariable, PolynomialVariable],
) -> GaussianRealificationResult:
    """Compute the real and imaginary parts of p(x+i y) via binomial expansion."""
    _admit_gaussian_realification(polynomial, target_variables)

    # Accumulate monomials for real and imag parts: dict (ix, iy) -> Fraction
    real_monomials: dict[tuple[int, int], Fraction] = {}
    imag_monomials: dict[tuple[int, int], Fraction] = {}

    for term in polynomial.terms:
        k = term.exponent
        a_real, a_imag = term.coefficient.as_fractions()
        # For each j in 0..k, binom(k,j) x^{k-j} (i y)^j
        # (i y)^j = i^j y^j, so separate real/imag based on j mod 4
        # We need to multiply by a = a_real + i a_imag, then separate.
        # (a_real + i a_imag) * binom * x^{k-j} * i^j y^j
        # = binom * x^{k-j} y^j * (a_real + i a_imag) * i^j
        # i^j cycles: 0->1,1->i,2->-1,3->-i
        for j in range(k + 1):
            coeff = Fraction(comb(k, j), 1)
            exp_x = k - j
            exp_y = j
            ij = j % 4
            # (i^j) = 1,i,-1,-i
            # Multiply (a_real + i a_imag) * i^j:
            # j0: (a_real + i a_imag)*1 = a_real + i a_imag => real a_real, imag a_imag
            # j1: *i => -a_imag + i a_real => real -a_imag, imag a_real
            # j2: *-1 => -a_real - i a_imag => real -a_real, imag -a_imag
            # j3: *-i => a_imag - i a_real => real a_imag, imag -a_real
            if ij == 0:
                r_coeff = a_real
                i_coeff = a_imag
            elif ij == 1:
                r_coeff = -a_imag
                i_coeff = a_real
            elif ij == 2:
                r_coeff = -a_real
                i_coeff = -a_imag
            else:  # 3
                r_coeff = a_imag
                i_coeff = -a_real
            # Multiply by binom
            r_contrib = coeff * r_coeff
            i_contrib = coeff * i_coeff
            key = (exp_x, exp_y)
            if r_contrib != 0:
                real_monomials[key] = (
                    real_monomials.get(key, Fraction(0, 1)) + r_contrib
                )
            if i_contrib != 0:
                imag_monomials[key] = (
                    imag_monomials.get(key, Fraction(0, 1)) + i_contrib
                )

    real_part = _build_rational_polynomial(target_variables, real_monomials)
    imag_part = _build_rational_polynomial(target_variables, imag_monomials)

    return GaussianRealificationResult._from_kernel(
        source_polynomial=polynomial,
        target_variables=target_variables,
        real_part=real_part,
        imag_part=imag_part,
    )


__all__ = [
    "GaussianComplexCoefficient",
    "GaussianRealificationRequest",
    "GaussianRealificationResult",
    "UnivariateGaussianPolynomial",
    "UnivariateGaussianPolynomialTerm",
    "gaussian_realification",
]
