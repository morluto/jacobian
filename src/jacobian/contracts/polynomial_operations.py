"""Contracts for exact polynomial invariants over ``QQ``."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian.contracts.exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalInteger,
    CanonicalRational,
)
from jacobian.contracts.results import ContractModel
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_TERMS,
    MAX_POLYNOMIAL_VARIABLES,
    PolynomialVariable,
    RationalPolynomial,
    require_polynomial_budget,
)

_MAX_COEFFICIENT_DIGITS = 256
_MAX_GCD_TERMS = 512
_MAX_INVARIANT_TERMS = 256
_MAX_GCD_DEGREE = 127
_MAX_ELIMINATION_DEGREE_SUM = 64
_MAX_DISCRIMINANT_DEGREE = 32
_MAX_SQUARE_FREE_EXPONENT = 64
_MAX_ELEMENTARY_DEGREE = 127
_MAX_INTEGER_COEFFICIENT_DIGITS = 256


def _degree(polynomial: RationalPolynomial, variable_index: int) -> int:
    return max(
        (term.exponents[variable_index] for term in polynomial.polynomial.terms),
        default=0,
    )


class PolynomialPairRequest(ContractModel):
    """Two polynomials in one identical declared rational polynomial ring."""

    left: RationalPolynomial
    right: RationalPolynomial

    @model_validator(mode="after")
    def require_matching_rings(self) -> Self:
        if self.left.variables != self.right.variables:
            raise ValueError("polynomials must use the same ordered variables")
        return self


class PolynomialGcdRequest(PolynomialPairRequest):
    @model_validator(mode="after")
    def require_univariate_budget(self) -> Self:
        if len(self.left.variables) != 1:
            raise ValueError("Bézout GCD currently supports one variable over QQ")
        for polynomial in (self.left, self.right):
            require_polynomial_budget(
                polynomial,
                maximum_terms=_MAX_GCD_TERMS,
                maximum_exponent=_MAX_GCD_DEGREE,
            )
        return self


class PolynomialBezoutIdentity(ContractModel):
    left_multiplier: RationalPolynomial
    right_multiplier: RationalPolynomial


class PolynomialGcdResult(ContractModel):
    gcd: RationalPolynomial
    bezout: PolynomialBezoutIdentity
    normalization: Literal["MONIC"] = "MONIC"


class PolynomialResultantRequest(PolynomialPairRequest):
    elimination_variable: PolynomialVariable

    @model_validator(mode="after")
    def require_elimination_budget(self) -> Self:
        if self.elimination_variable not in self.left.variables:
            raise ValueError("elimination variable must belong to the declared ring")
        for polynomial in (self.left, self.right):
            require_polynomial_budget(
                polynomial,
                maximum_terms=_MAX_INVARIANT_TERMS,
                maximum_exponent=_MAX_ELIMINATION_DEGREE_SUM,
            )
        variable_index = self.left.variables.index(self.elimination_variable)
        degree_sum = _degree(self.left, variable_index) + _degree(
            self.right, variable_index
        )
        if degree_sum > _MAX_ELIMINATION_DEGREE_SUM:
            raise ValueError("Sylvester degree exceeds the resultant budget")
        return self


class PolynomialDiscriminantRequest(ContractModel):
    polynomial: RationalPolynomial
    variable: PolynomialVariable

    @model_validator(mode="after")
    def require_discriminant_budget(self) -> Self:
        if self.variable not in self.polynomial.variables:
            raise ValueError("discriminant variable must belong to the declared ring")
        require_polynomial_budget(
            self.polynomial,
            maximum_terms=_MAX_INVARIANT_TERMS,
            maximum_exponent=_MAX_SQUARE_FREE_EXPONENT,
        )
        variable_index = self.polynomial.variables.index(self.variable)
        if _degree(self.polynomial, variable_index) > _MAX_DISCRIMINANT_DEGREE:
            raise ValueError("main-variable degree exceeds the discriminant budget")
        return self


class PolynomialScalarValue(ContractModel):
    kind: Literal["SCALAR"] = "SCALAR"
    value: CanonicalRational


class PolynomialValue(ContractModel):
    kind: Literal["POLYNOMIAL"] = "POLYNOMIAL"
    value: RationalPolynomial


PolynomialInvariantValue = Annotated[
    PolynomialScalarValue | PolynomialValue,
    Field(discriminator="kind"),
]


class PolynomialResultantResult(ContractModel):
    elimination_variable: PolynomialVariable
    resultant: PolynomialInvariantValue
    convention: Literal["SYLVESTER_DETERMINANT"] = "SYLVESTER_DETERMINANT"


class PolynomialDiscriminantResult(ContractModel):
    variable: PolynomialVariable
    discriminant: PolynomialInvariantValue
    convention: Literal["STANDARD_UNIVARIATE"] = "STANDARD_UNIVARIATE"


class PolynomialSquareFreeRequest(ContractModel):
    polynomial: RationalPolynomial

    @model_validator(mode="after")
    def require_square_free_budget(self) -> Self:
        require_polynomial_budget(
            self.polynomial,
            maximum_terms=_MAX_GCD_TERMS,
            maximum_exponent=_MAX_SQUARE_FREE_EXPONENT,
        )
        return self


class PolynomialSquareFreeFactor(ContractModel):
    factor: RationalPolynomial
    multiplicity: int = Field(ge=1, le=_MAX_SQUARE_FREE_EXPONENT)


class PolynomialSquareFreeDecompositionResult(ContractModel):
    coefficient: CanonicalRational
    factors: tuple[PolynomialSquareFreeFactor, ...] = Field(max_length=64)
    reconstructed: RationalPolynomial
    normalization: Literal["MONIC_FACTORS"] = "MONIC_FACTORS"

    @model_validator(mode="after")
    def require_canonical_factor_records(self) -> Self:
        multiplicities = tuple(factor.multiplicity for factor in self.factors)
        if multiplicities != tuple(sorted(multiplicities)):
            raise ValueError("square-free factors must be ordered by multiplicity")
        if len(set(multiplicities)) != len(multiplicities):
            raise ValueError("each multiplicity must have one square-free factor")
        if any(
            factor.factor.variables != self.reconstructed.variables
            for factor in self.factors
        ):
            raise ValueError("square-free factors must use the source ring")
        return self


class PolynomialFactorRequest(ContractModel):
    """Univariate factorization request over ``QQ``."""

    polynomial: RationalPolynomial

    @model_validator(mode="after")
    def require_univariate_factor_budget(self) -> Self:
        if len(self.polynomial.variables) != 1:
            raise ValueError("factorization currently supports one variable over QQ")
        require_polynomial_budget(
            self.polynomial,
            maximum_terms=_MAX_GCD_TERMS,
            maximum_exponent=_MAX_GCD_DEGREE,
        )
        return self


class PolynomialIrreducibleFactor(ContractModel):
    factor: RationalPolynomial
    multiplicity: int = Field(ge=1, le=_MAX_GCD_DEGREE)


class PolynomialFactorizationResult(ContractModel):
    coefficient: CanonicalRational
    factors: tuple[PolynomialIrreducibleFactor, ...] = Field(max_length=64)
    reconstructed: RationalPolynomial
    normalization: Literal["CONTENT_AND_MONIC_IRREDUCIBLES"] = (
        "CONTENT_AND_MONIC_IRREDUCIBLES"
    )
    product_reconstruction: Literal["EXACT"] = "EXACT"

    @model_validator(mode="after")
    def require_canonical_irreducible_records(self) -> Self:
        if any(
            factor.factor.variables != self.reconstructed.variables
            for factor in self.factors
        ):
            raise ValueError("irreducible factors must use the source ring")
        ordered = tuple(
            sorted(
                self.factors,
                key=lambda record: (
                    record.multiplicity,
                    max(
                        (
                            sum(term.exponents)
                            for term in record.factor.polynomial.terms
                        ),
                        default=0,
                    ),
                    tuple(
                        (
                            term.exponents,
                            term.coefficient.num,
                            term.coefficient.den,
                        )
                        for term in record.factor.polynomial.terms
                    ),
                ),
            )
        )
        if self.factors != ordered:
            raise ValueError(
                "irreducible factors must be ordered by multiplicity, degree, "
                "and sparse term fingerprint"
            )
        return self


class PolynomialGroebnerBudget(ContractModel):
    """Enforced wall and result limits for one isolated Gröbner computation."""

    wall_seconds: StrictInt = Field(default=10, ge=1, le=60)
    maximum_basis_polynomials: StrictInt = Field(default=64, ge=1, le=64)
    maximum_output_terms: StrictInt = Field(default=1024, ge=1, le=1024)


class PolynomialGroebnerBasisRequest(ContractModel):
    generators: tuple[RationalPolynomial, ...] = Field(min_length=1, max_length=16)
    monomial_order: Literal["lex", "grlex", "grevlex"] = "grevlex"
    resource_budget: PolynomialGroebnerBudget = Field(
        default_factory=PolynomialGroebnerBudget
    )

    @model_validator(mode="after")
    def require_groebner_budget(self) -> Self:
        variables = self.generators[0].variables
        if any(generator.variables != variables for generator in self.generators):
            raise ValueError("all ideal generators must use the same ordered ring")
        if sum(len(generator.polynomial.terms) for generator in self.generators) > 256:
            raise ValueError("ideal generators exceed the aggregate term budget")
        for generator in self.generators:
            require_polynomial_budget(
                generator,
                maximum_terms=MAX_POLYNOMIAL_TERMS,
                maximum_exponent=12,
                maximum_coefficient_digits=128,
                label="ideal generator",
            )
            if any(sum(term.exponents) > 12 for term in generator.polynomial.terms):
                raise ValueError("ideal generator exceeds total degree 12")
        return self


class PolynomialGroebnerBasisResult(ContractModel):
    variables: tuple[PolynomialVariable, ...] = Field(
        min_length=1,
        max_length=MAX_POLYNOMIAL_VARIABLES,
    )
    monomial_order: Literal["lex", "grlex", "grevlex"]
    basis: tuple[RationalPolynomial, ...] = Field(max_length=64)
    completion: Literal["COMPLETE"] = "COMPLETE"
    normalization: Literal["REDUCED_MONIC"] = "REDUCED_MONIC"

    @model_validator(mode="after")
    def require_canonical_basis_ring(self) -> Self:
        if any(polynomial.variables != self.variables for polynomial in self.basis):
            raise ValueError("every basis polynomial must use the declared ring")
        if sum(len(polynomial.polynomial.terms) for polynomial in self.basis) > 1024:
            raise ValueError("Gröbner basis exceeds the aggregate output term limit")
        return self


class IntegerPolynomial(ContractModel):
    """Canonical dense polynomial in ``ZZ[x]``, highest degree first."""

    coefficient_order: Literal["DESCENDING_DEGREE"] = "DESCENDING_DEGREE"
    coefficients: tuple[CanonicalInteger, ...] = Field(
        min_length=1,
        max_length=MAX_POLYNOMIAL_TERMS,
    )

    @model_validator(mode="after")
    def require_canonical_coefficients(self) -> Self:
        if len(self.coefficients) > 1 and self.coefficients[0] == "0":
            raise ValueError("leading zero coefficients must be omitted")
        if any(
            len(coefficient.lstrip("-")) > MAX_CANONICAL_RATIONAL_DIGITS
            for coefficient in self.coefficients
        ):
            raise ValueError(
                "integer coefficient exceeds the shared representation limit"
            )
        return self


def _require_integer_polynomial_budget(polynomial: IntegerPolynomial) -> None:
    if len(polynomial.coefficients) > _MAX_ELEMENTARY_DEGREE + 1:
        raise ValueError("integer polynomial exceeds the degree-127 operation budget")
    if any(
        len(coefficient.lstrip("-")) > _MAX_INTEGER_COEFFICIENT_DIGITS
        for coefficient in polynomial.coefficients
    ):
        raise ValueError("integer coefficient exceeds the decimal-digit budget")


class IntegerPolynomialRequest(ContractModel):
    polynomial: IntegerPolynomial

    @model_validator(mode="after")
    def require_operation_budget(self) -> Self:
        _require_integer_polynomial_budget(self.polynomial)
        return self


class IntegerPolynomialShiftRequest(IntegerPolynomialRequest):
    shift: StrictInt = Field(ge=-10_000, le=10_000)


class IntegerPolynomialShiftResult(ContractModel):
    shift: StrictInt = Field(ge=-10_000, le=10_000)
    shifted: IntegerPolynomial
    convention: Literal["SUBSTITUTE_X_PLUS_SHIFT"] = "SUBSTITUTE_X_PLUS_SHIFT"


class IntegerPolynomialPairRequest(ContractModel):
    left: IntegerPolynomial
    right: IntegerPolynomial

    @model_validator(mode="after")
    def require_operation_budget(self) -> Self:
        _require_integer_polynomial_budget(self.left)
        _require_integer_polynomial_budget(self.right)
        return self


class IntegerPolynomialGcdResult(ContractModel):
    gcd: IntegerPolynomial
    left_content: CanonicalInteger
    right_content: CanonicalInteger
    gcd_content: CanonicalInteger
    normalization: Literal["NONNEGATIVE_LEADING_COEFFICIENT"] = (
        "NONNEGATIVE_LEADING_COEFFICIENT"
    )


class IntegerPolynomialContentResult(ContractModel):
    content: CanonicalInteger
    convention: Literal["NONNEGATIVE_COEFFICIENT_GCD"] = "NONNEGATIVE_COEFFICIENT_GCD"


class IntegerPolynomialPrimitivePartResult(ContractModel):
    content: CanonicalInteger
    primitive_part: IntegerPolynomial
    reconstruction: IntegerPolynomial
    convention: Literal["NONNEGATIVE_CONTENT"] = "NONNEGATIVE_CONTENT"


class IntegerPolynomialEvaluationRequest(IntegerPolynomialRequest):
    point: CanonicalInteger

    @model_validator(mode="after")
    def require_bounded_point(self) -> Self:
        if len(self.point.lstrip("-")) > _MAX_INTEGER_COEFFICIENT_DIGITS:
            raise ValueError("evaluation point exceeds the decimal-digit budget")
        return self


class IntegerPolynomialEvaluationResult(ContractModel):
    point: CanonicalInteger
    value: CanonicalInteger


class IntegerPolynomialCompositionRequest(ContractModel):
    outer: IntegerPolynomial
    inner: IntegerPolynomial

    @model_validator(mode="after")
    def require_bounded_output_degree(self) -> Self:
        _require_integer_polynomial_budget(self.outer)
        _require_integer_polynomial_budget(self.inner)
        outer_degree = len(self.outer.coefficients) - 1
        inner_degree = len(self.inner.coefficients) - 1
        if outer_degree * inner_degree > _MAX_ELEMENTARY_DEGREE:
            raise ValueError("composition exceeds the degree-127 output budget")
        return self


class IntegerPolynomialCompositionResult(ContractModel):
    composition: IntegerPolynomial


class RationalPolynomialRequest(ContractModel):
    polynomial: RationalPolynomial

    @model_validator(mode="after")
    def require_univariate_budget(self) -> Self:
        if len(self.polynomial.variables) != 1:
            raise ValueError("elementary polynomial operations require one variable")
        require_polynomial_budget(
            self.polynomial,
            maximum_terms=_MAX_GCD_TERMS,
            maximum_exponent=_MAX_ELEMENTARY_DEGREE,
        )
        return self


class RationalPolynomialDivisionRequest(PolynomialPairRequest):
    @model_validator(mode="after")
    def require_division_budget(self) -> Self:
        if len(self.left.variables) != 1:
            raise ValueError("polynomial division requires one variable")
        if not self.right.polynomial.terms:
            raise ValueError("divisor polynomial must be nonzero")
        for polynomial in (self.left, self.right):
            require_polynomial_budget(
                polynomial,
                maximum_terms=_MAX_GCD_TERMS,
                maximum_exponent=_MAX_ELEMENTARY_DEGREE,
            )
        return self


class RationalPolynomialDivisionResult(ContractModel):
    quotient: RationalPolynomial
    remainder: RationalPolynomial
    reconstruction: RationalPolynomial


class RationalPolynomialEvaluationRequest(RationalPolynomialRequest):
    point: CanonicalRational


class RationalPolynomialEvaluationResult(ContractModel):
    point: CanonicalRational
    value: CanonicalRational


class RationalPolynomialDerivativeResult(ContractModel):
    derivative: RationalPolynomial


class RationalPolynomialIntegralResult(ContractModel):
    antiderivative: RationalPolynomial
    integration_constant: Literal["ZERO"] = "ZERO"


class RationalFunctionRequest(ContractModel):
    numerator: RationalPolynomial
    denominator: RationalPolynomial

    @model_validator(mode="after")
    def require_matching_univariate_ring_and_budget(self) -> Self:
        if self.numerator.variables != self.denominator.variables:
            raise ValueError("numerator and denominator must use the same ring")
        if len(self.numerator.variables) != 1:
            raise ValueError("partial fractions require one variable")
        if not self.denominator.polynomial.terms:
            raise ValueError("denominator polynomial must be nonzero")
        for polynomial in (self.numerator, self.denominator):
            require_polynomial_budget(
                polynomial,
                maximum_terms=_MAX_INVARIANT_TERMS,
                maximum_exponent=_MAX_ELEMENTARY_DEGREE,
            )
        return self


class RationalPartialFractionTerm(ContractModel):
    numerator: RationalPolynomial
    denominator_factor: RationalPolynomial
    denominator_exponent: int = Field(ge=1, le=_MAX_ELEMENTARY_DEGREE)


class RationalPartialFractionResult(ContractModel):
    polynomial_part: RationalPolynomial
    terms: tuple[RationalPartialFractionTerm, ...] = Field(max_length=128)
    reconstruction_numerator: RationalPolynomial
    reconstruction_denominator: RationalPolynomial
    decomposition_field: Literal["QQ"] = "QQ"


__all__ = [
    "IntegerPolynomial",
    "IntegerPolynomialCompositionRequest",
    "IntegerPolynomialCompositionResult",
    "IntegerPolynomialContentResult",
    "IntegerPolynomialEvaluationRequest",
    "IntegerPolynomialEvaluationResult",
    "IntegerPolynomialGcdResult",
    "IntegerPolynomialPairRequest",
    "IntegerPolynomialPrimitivePartResult",
    "IntegerPolynomialRequest",
    "PolynomialBezoutIdentity",
    "PolynomialDiscriminantRequest",
    "PolynomialDiscriminantResult",
    "PolynomialFactorRequest",
    "PolynomialFactorizationResult",
    "PolynomialGcdRequest",
    "PolynomialGcdResult",
    "PolynomialGroebnerBasisRequest",
    "PolynomialGroebnerBasisResult",
    "PolynomialGroebnerBudget",
    "PolynomialInvariantValue",
    "PolynomialIrreducibleFactor",
    "PolynomialPairRequest",
    "PolynomialResultantRequest",
    "PolynomialResultantResult",
    "PolynomialScalarValue",
    "PolynomialSquareFreeDecompositionResult",
    "PolynomialSquareFreeFactor",
    "PolynomialSquareFreeRequest",
    "PolynomialValue",
    "RationalFunctionRequest",
    "RationalPartialFractionResult",
    "RationalPartialFractionTerm",
    "RationalPolynomialDerivativeResult",
    "RationalPolynomialDivisionRequest",
    "RationalPolynomialDivisionResult",
    "RationalPolynomialEvaluationRequest",
    "RationalPolynomialEvaluationResult",
    "RationalPolynomialIntegralResult",
    "RationalPolynomialRequest",
]
