"""Contracts for exact polynomial invariants over ``QQ``."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any, Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalInteger,
    CanonicalRational,
)
from jacobian._models import StrictModel
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_TERMS,
    MAX_POLYNOMIAL_VARIABLES,
    PolynomialVariable,
    RationalPolynomial,
    require_polynomial_budget,
)

_MAX_COEFFICIENT_DIGITS = 256
_MAX_GCD_TERMS = 1024
_MAX_INVARIANT_TERMS = 256
_MAX_GCD_DEGREE = 500
_MAX_ELIMINATION_DEGREE_SUM = 128
_MAX_DISCRIMINANT_DEGREE = 64
_MAX_SQUARE_FREE_EXPONENT = 64
_MAX_ELEMENTARY_DEGREE = 127
_MAX_INTEGER_COEFFICIENT_DIGITS = 256


def _degree(polynomial: RationalPolynomial, variable_index: int) -> int:
    return max(
        (term.exponents[variable_index] for term in polynomial.polynomial.terms),
        default=0,
    )


def _polynomial_total_degree(polynomial: RationalPolynomial) -> int:
    return max(
        (sum(term.exponents) for term in polynomial.polynomial.terms),
        default=0,
    )


class PolynomialPairRequest(StrictModel):
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

    @model_validator(mode="after")
    def require_not_both_zero(self) -> Self:
        """Reject gcd(0, 0): zero has no monic normalization."""
        left_zero = len(self.left.polynomial.terms) == 0
        right_zero = len(self.right.polynomial.terms) == 0
        if left_zero and right_zero:
            raise ValueError("gcd(0, 0) is undefined: zero has no monic normalization")
        return self


class PolynomialBezoutIdentity(StrictModel):
    left_multiplier: RationalPolynomial
    right_multiplier: RationalPolynomial


class PolynomialGcdResult(StrictModel):
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


class PolynomialDiscriminantRequest(StrictModel):
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


class PolynomialScalarValue(StrictModel):
    kind: Literal["SCALAR"] = "SCALAR"
    value: CanonicalRational


class PolynomialValue(StrictModel):
    kind: Literal["POLYNOMIAL"] = "POLYNOMIAL"
    value: RationalPolynomial


PolynomialInvariantValue = Annotated[
    PolynomialScalarValue | PolynomialValue,
    Field(discriminator="kind"),
]


class PolynomialResultantResult(StrictModel):
    elimination_variable: PolynomialVariable
    resultant: PolynomialInvariantValue
    convention: Literal["SYLVESTER_DETERMINANT"] = "SYLVESTER_DETERMINANT"


class PolynomialDiscriminantResult(StrictModel):
    variable: PolynomialVariable
    discriminant: PolynomialInvariantValue
    convention: Literal["STANDARD_UNIVARIATE"] = "STANDARD_UNIVARIATE"


class PolynomialSquareFreeRequest(StrictModel):
    polynomial: RationalPolynomial

    @model_validator(mode="after")
    def require_square_free_budget(self) -> Self:
        require_polynomial_budget(
            self.polynomial,
            maximum_terms=_MAX_GCD_TERMS,
            maximum_exponent=_MAX_SQUARE_FREE_EXPONENT,
        )
        return self


class PolynomialSquareFreeFactor(StrictModel):
    factor: RationalPolynomial
    multiplicity: int = Field(ge=1, le=_MAX_SQUARE_FREE_EXPONENT)


class PolynomialSquareFreeDecompositionResult(StrictModel):
    """The square-free decomposition bound to its source polynomial.

    Retains the canonical source polynomial so validation replays
    ``reconstructed = polynomial`` and authenticates the defining relation
    ``polynomial = coefficient * product(factor^multiplicity)`` together
    with the uniqueness of the monic square-free parts: an exact product
    at distinct multiplicities does not force pairwise-coprime square-free
    factors, so validation recomputes the canonical monic decomposition of
    the retained source with the same bounded backend invocation the
    operation itself performs and compares the coefficient and the
    multiplicity-weighted monic records against the claim for every ring
    arity. Neither arity materializes a claimed product intermediate.
    """

    polynomial: RationalPolynomial
    coefficient: CanonicalRational
    factors: tuple[PolynomialSquareFreeFactor, ...] = Field(max_length=64)
    reconstructed: RationalPolynomial
    normalization: Literal["MONIC_FACTORS"] = "MONIC_FACTORS"

    @model_validator(mode="after")
    def require_canonical_factor_records(self) -> Self:
        from jacobian.math.polynomials._conversions import (
            rational_polynomial_to_sympy,
        )
        from jacobian.math.polynomials._sympy import (
            polynomial_square_free_decomposition,
        )

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
        require_polynomial_budget(
            self.polynomial,
            maximum_terms=_MAX_GCD_TERMS,
            maximum_exponent=_MAX_SQUARE_FREE_EXPONENT,
            maximum_coefficient_digits=MAX_CANONICAL_RATIONAL_DIGITS,
            label="retained source polynomial",
        )
        require_polynomial_budget(
            self.reconstructed,
            maximum_terms=_MAX_GCD_TERMS,
            maximum_exponent=_MAX_SQUARE_FREE_EXPONENT,
            maximum_coefficient_digits=MAX_CANONICAL_RATIONAL_DIGITS,
            label="reconstructed polynomial",
        )
        source = rational_polynomial_to_sympy(self.polynomial)
        if rational_polynomial_to_sympy(self.reconstructed) != source:
            raise ValueError("reconstructed must equal the retained source polynomial")
        _verify_exact_factor_product(
            source,
            self.factors,
            coefficient=self.coefficient,
            mismatch_message=(
                "square-free factors must reconstruct the retained source "
                "polynomial exactly"
            ),
            label="square-free",
            maximum_exponent=_MAX_SQUARE_FREE_EXPONENT,
            replay_decomposition=polynomial_square_free_decomposition,
        )
        return self


class PolynomialFactorRequest(StrictModel):
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


class PolynomialIrreducibleFactor(StrictModel):
    factor: RationalPolynomial
    multiplicity: int = Field(ge=1, le=_MAX_GCD_DEGREE)


class PolynomialFactorizationResult(StrictModel):
    """The exact univariate factorization bound to its source polynomial.

    Retains the canonical source polynomial so validation replays the
    defining relations ``reconstructed = polynomial`` and
    ``polynomial = coefficient * product(factor^multiplicity)`` together
    with the uniqueness of the content-and-monic-irreducibles
    normalization: an exact product does not force irreducible records, so
    validation recomputes the canonical factorization of the retained
    source with the same bounded backend invocation the operation itself
    performs and compares the coefficient and the multiplicity-weighted
    monic records against the claim.  The literal
    ``product_reconstruction = EXACT`` label is derived from that replay,
    never accepted as evidence; no claimed product intermediate is ever
    materialized.
    """

    polynomial: RationalPolynomial
    coefficient: CanonicalRational
    factors: tuple[PolynomialIrreducibleFactor, ...] = Field(max_length=64)
    reconstructed: RationalPolynomial
    normalization: Literal["CONTENT_AND_MONIC_IRREDUCIBLES"] = (
        "CONTENT_AND_MONIC_IRREDUCIBLES"
    )
    product_reconstruction: Literal["EXACT"] = "EXACT"

    @model_validator(mode="after")
    def require_canonical_irreducible_records(self) -> Self:
        from jacobian.math.polynomials._conversions import (
            rational_polynomial_to_sympy,
        )
        from jacobian.math.polynomials._sympy import polynomial_factorization

        if len(self.polynomial.variables) != 1:
            raise ValueError("factorization currently supports one variable over QQ")
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
        require_polynomial_budget(
            self.polynomial,
            maximum_terms=_MAX_GCD_TERMS,
            maximum_exponent=_MAX_GCD_DEGREE,
            maximum_coefficient_digits=MAX_CANONICAL_RATIONAL_DIGITS,
            label="retained source polynomial",
        )
        require_polynomial_budget(
            self.reconstructed,
            maximum_terms=_MAX_GCD_TERMS,
            maximum_exponent=_MAX_GCD_DEGREE,
            maximum_coefficient_digits=MAX_CANONICAL_RATIONAL_DIGITS,
            label="reconstructed polynomial",
        )
        source = rational_polynomial_to_sympy(self.polynomial)
        if rational_polynomial_to_sympy(self.reconstructed) != source:
            raise ValueError("reconstructed must equal the retained source polynomial")
        _verify_exact_factor_product(
            source,
            self.factors,
            coefficient=self.coefficient,
            mismatch_message=(
                "factorization must reconstruct the retained source polynomial exactly"
            ),
            label="irreducible",
            maximum_exponent=_MAX_GCD_DEGREE,
            replay_decomposition=polynomial_factorization,
        )
        return self


def _canonical_poly_key(polynomial: Any) -> Any:
    """Return the hashable canonical form of one monic QQ ``Poly``."""

    return tuple(
        sorted(
            (monom, int(coeff.p), int(coeff.q)) for monom, coeff in polynomial.terms()
        )
    )


def _verify_exact_factor_product(
    source: Any,
    factors: tuple[PolynomialSquareFreeFactor, ...]
    | tuple[PolynomialIrreducibleFactor, ...],
    *,
    coefficient: CanonicalRational,
    mismatch_message: str,
    label: str,
    maximum_exponent: int,
    replay_decomposition: Callable[[Any], tuple[Any, tuple[tuple[Any, int], ...], Any]],
) -> None:
    """Replay ``coefficient * product(factor ** multiplicity) == source``.

    Over ``QQ`` (characteristic zero) both the content-and-monic-irreducibles
    factorization and the monic square-free decomposition are unique, so the
    defining product identity holds exactly when the claimed content and the
    claimed multiplicity-weighted monic factor records equal the ones the
    maintained backend re-derives from the retained source polynomial: the
    replay invokes the same bounded decomposition entry points the producing
    operation itself ran on this identical admitted source envelope and
    compares content and records.  No lane ever divides or forms a claimed
    product, quotient, or power, so no claim can force quotient expansion.

    Verifying only the product is not enough to authenticate either
    contract.  A product-exact claim whose parts overlap — ``(x - 1)`` and
    ``((x - 1)(x + 1))**2`` against ``(x - 1)**3(x + 1)**2`` — survives
    every exact division at distinct multiplicities while grouping
    irreducibles into parts that are neither pairwise coprime nor the
    square-free decomposition, and a single reducible record such as
    ``x**2 - 1`` reconstructs ``x**2 - 1`` without being an irreducible
    factorization.  Division-based lanes also cannot be bounded in general
    no matter how they are preflighted: the exact quotient of two admitted
    sparse sources can densify combinatorially beyond the representation
    envelope, and an inexact division builds partial quotients unconstrained
    by any per-variable degree box before it discovers non-divisibility.
    Canonical recomparison rejects all of these typedly with zero divisions.
    """

    from sympy import Rational

    from jacobian.math.polynomials._conversions import (
        rational_polynomial_to_sympy,
    )

    if coefficient.as_fraction() == 0 and factors:
        raise ValueError("results with zero content retain no factors")
    for record in factors:
        require_polynomial_budget(
            record.factor,
            maximum_terms=_MAX_GCD_TERMS,
            maximum_exponent=maximum_exponent,
            maximum_coefficient_digits=MAX_CANONICAL_RATIONAL_DIGITS,
            label=f"{label} factor",
        )
        if _polynomial_total_degree(record.factor) == 0:
            raise ValueError(f"{label} factor must be non-constant")
    try:
        replayed_coefficient, replayed_factors, _ = replay_decomposition(source)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"invalid {label} factor replay") from exc
    if Rational(*coefficient.as_integer_ratio()) != replayed_coefficient:
        raise ValueError(mismatch_message)
    claimed: dict[Any, int] = {}
    for record in factors:
        key = _canonical_poly_key(rational_polynomial_to_sympy(record.factor))
        claimed[key] = claimed.get(key, 0) + record.multiplicity
    replayed: dict[Any, int] = {}
    for factor, multiplicity in replayed_factors:
        key = _canonical_poly_key(factor)
        replayed[key] = replayed.get(key, 0) + multiplicity
    if claimed != replayed:
        raise ValueError(mismatch_message)


class PolynomialGroebnerBudget(StrictModel):
    """Enforced wall and result limits for one isolated Gröbner computation."""

    wall_seconds: StrictInt = Field(default=10, ge=1, le=60)
    maximum_basis_polynomials: StrictInt = Field(default=64, ge=1, le=64)
    maximum_output_terms: StrictInt = Field(default=1024, ge=1, le=1024)


class PolynomialGroebnerBasisRequest(StrictModel):
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


class PolynomialGroebnerBasisResult(StrictModel):
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


class IntegerPolynomial(StrictModel):
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


class IntegerPolynomialRequest(StrictModel):
    polynomial: IntegerPolynomial

    @model_validator(mode="after")
    def require_operation_budget(self) -> Self:
        _require_integer_polynomial_budget(self.polynomial)
        return self


class IntegerPolynomialShiftRequest(IntegerPolynomialRequest):
    shift: StrictInt = Field(ge=-10_000, le=10_000)


class IntegerPolynomialShiftResult(StrictModel):
    shift: StrictInt = Field(ge=-10_000, le=10_000)
    shifted: IntegerPolynomial
    convention: Literal["SUBSTITUTE_X_PLUS_SHIFT"] = "SUBSTITUTE_X_PLUS_SHIFT"


class IntegerPolynomialPairRequest(StrictModel):
    left: IntegerPolynomial
    right: IntegerPolynomial

    @model_validator(mode="after")
    def require_operation_budget(self) -> Self:
        _require_integer_polynomial_budget(self.left)
        _require_integer_polynomial_budget(self.right)
        return self


class IntegerPolynomialGcdResult(StrictModel):
    gcd: IntegerPolynomial
    left_content: CanonicalInteger
    right_content: CanonicalInteger
    gcd_content: CanonicalInteger
    normalization: Literal["NONNEGATIVE_LEADING_COEFFICIENT"] = (
        "NONNEGATIVE_LEADING_COEFFICIENT"
    )


class IntegerPolynomialContentResult(StrictModel):
    content: CanonicalInteger
    convention: Literal["NONNEGATIVE_COEFFICIENT_GCD"] = "NONNEGATIVE_COEFFICIENT_GCD"


class IntegerPolynomialPrimitivePartResult(StrictModel):
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


class IntegerPolynomialEvaluationResult(StrictModel):
    point: CanonicalInteger
    value: CanonicalInteger


class IntegerPolynomialCompositionRequest(StrictModel):
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


class IntegerPolynomialCompositionResult(StrictModel):
    composition: IntegerPolynomial


class RationalPolynomialRequest(StrictModel):
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


class RationalPolynomialDivisionResult(StrictModel):
    quotient: RationalPolynomial
    remainder: RationalPolynomial
    reconstruction: RationalPolynomial


class RationalPolynomialEvaluationRequest(RationalPolynomialRequest):
    point: CanonicalRational


class RationalPolynomialEvaluationResult(StrictModel):
    point: CanonicalRational
    value: CanonicalRational


class RationalPolynomialDerivativeResult(StrictModel):
    derivative: RationalPolynomial


class RationalPolynomialIntegralResult(StrictModel):
    antiderivative: RationalPolynomial
    integration_constant: Literal["ZERO"] = "ZERO"


class RationalFunctionRequest(StrictModel):
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


class RationalPartialFractionTerm(StrictModel):
    numerator: RationalPolynomial
    denominator_factor: RationalPolynomial
    denominator_exponent: int = Field(ge=1, le=_MAX_ELEMENTARY_DEGREE)


class RationalPartialFractionResult(StrictModel):
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
