"""Contracts for exact polynomial invariants over ``QQ``."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalInteger,
    CanonicalRational,
)
from jacobian._models import StrictModel
from jacobian.math.polynomials._replay import (
    generators_reduce_to_zero,
    remainder_matches_claim,
    replayed_remainder_exceeds_budget,
    retained_basis_is_groebner,
    retained_source_basis_exceeds_budget,
)
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_TERMS,
    MAX_POLYNOMIAL_VARIABLES,
    PolynomialVariable,
    RationalPolynomial,
    RationalPolynomialIdeal,
    require_polynomial_budget,
)

_MAX_RESULT_POLYNOMIAL_TERMS = 1_024
"""Exact-result term boundary shared by the polynomial output converters."""

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


_IDEAL_GENERATOR_COUNT_LIMIT = 16
"""Generators admitted per ideal-membership request, matching the description."""

_IDEAL_GENERATOR_TOTAL_DEGREE = 12
"""Total-degree cap per ideal generator term; subsumes each variable's exponent."""


def _require_ideal_generator_budget(ideal: RationalPolynomialIdeal) -> None:
    """Apply the advertised per-generator admission limits."""

    if len(ideal.generators) > _IDEAL_GENERATOR_COUNT_LIMIT:
        raise ValueError(
            f"ideal exceeds the {_IDEAL_GENERATOR_COUNT_LIMIT}-generator operation budget"
        )
    if sum(len(gen.polynomial.terms) for gen in ideal.generators) > 256:
        raise ValueError("ideal generators exceed the aggregate term budget")
    for gen in ideal.generators:
        require_polynomial_budget(
            gen,
            maximum_terms=MAX_POLYNOMIAL_TERMS,
            maximum_exponent=_IDEAL_GENERATOR_TOTAL_DEGREE,
            maximum_coefficient_digits=128,
            label="ideal generator",
        )
        for term in gen.polynomial.terms:
            if sum(term.exponents) > _IDEAL_GENERATOR_TOTAL_DEGREE:
                raise ValueError(
                    "ideal generator degree exceeds the "
                    f"{_IDEAL_GENERATOR_TOTAL_DEGREE}-total-degree operation budget"
                )


def _require_queried_polynomial_budget(polynomial: RationalPolynomial) -> None:
    """Apply the advertised queried-polynomial admission limits.

    The normal form of the zero ideal is the input polynomial itself;
    cap the input at the 1,024-term result boundary so an accepted
    request can never leak a result-budget host exception.
    """

    if len(polynomial.polynomial.terms) > _MAX_RESULT_POLYNOMIAL_TERMS:
        raise ValueError("polynomial exceeds the 1,024-term exact-result limit")
    require_polynomial_budget(
        polynomial,
        maximum_terms=MAX_POLYNOMIAL_TERMS,
        maximum_exponent=12,
        maximum_coefficient_digits=128,
        label="polynomial",
    )
    # The advertised work domain bounds per-term total degree, not just
    # each individual exponent.
    for term in polynomial.polynomial.terms:
        if sum(term.exponents) > _IDEAL_GENERATOR_TOTAL_DEGREE:
            raise ValueError(
                "polynomial degree exceeds the "
                f"{_IDEAL_GENERATOR_TOTAL_DEGREE}-total-degree operation budget"
            )


def _validate_membership_source(
    ideal: RationalPolynomialIdeal, polynomial: RationalPolynomial
) -> None:
    """Apply the advertised ideal-membership work budgets to source values."""

    variables = ideal.variables
    if any(gen.variables != variables for gen in ideal.generators):
        raise ValueError("all ideal generators must use the same ordered ring")
    if polynomial.variables != variables:
        raise ValueError("polynomial must use the same ordered ring as the ideal")
    _require_ideal_generator_budget(ideal)
    _require_queried_polynomial_budget(polynomial)


def _validate_retained_basis(
    groebner_basis: tuple[RationalPolynomial, ...] | None,
    variables: tuple[PolynomialVariable, ...],
) -> None:
    """Check a retained Gröbner basis stays inside the aggregate output budget."""

    if groebner_basis is None:
        return
    if not groebner_basis:
        # Empty basis is valid only for the zero ideal; the per-result
        # validators enforce the correct correlation via the replay checks.
        return
    if any(element.variables != variables for element in groebner_basis):
        raise ValueError("every retained basis polynomial must use the declared ring")
    if any(not element.polynomial.terms for element in groebner_basis):
        raise ValueError("retained Gröbner basis must contain nonzero polynomials only")
    if sum(len(element.polynomial.terms) for element in groebner_basis) > 1024:
        raise ValueError(
            "retained Gröbner basis exceeds the aggregate output term limit"
        )


class IdealMembershipRequest(StrictModel):
    """Decide membership of one polynomial in a bounded polynomial ideal.

    Accepts the domain-owned canonical ``RationalPolynomialIdeal`` value so
    serialized ideals compose without unpacking.  The ideal is bounded to 16
    generators with total degree at most 12 and the request polynomial to
    1,024 terms, per-term total degree at most 12, and coefficient
    components of at most 128 digits, so the bounded backend work cannot
    expand into an unrepresentable exact result without the typed budget
    outcome.
    """

    ideal: RationalPolynomialIdeal = Field(
        description=(
            "A bounded polynomial ideal in one ordered QQ ring: at most 16 "
            "generators with at most 256 aggregate terms, total degree at "
            "most 12 per generator term, and coefficient components at most "
            "128 digits."
        ),
    )
    polynomial: RationalPolynomial = Field(
        description=(
            "The queried polynomial in the ideal's ring: at most 1,024 "
            "terms, per-term total degree at most 12, and coefficient "
            "components of at most 128 digits; requests outside these "
            "operation-specific limits are rejected."
        ),
    )
    monomial_order: Literal["lex", "grlex", "grevlex"] = "grevlex"

    @model_validator(mode="after")
    def require_matching_rings(self) -> Self:
        _validate_membership_source(self.ideal, self.polynomial)
        return self


def _require_authentic_retained_basis(
    ideal: RationalPolynomialIdeal,
    groebner_basis: tuple[RationalPolynomial, ...],
    monomial_order: str,
) -> None:
    """Check the retained basis spans the ideal and is its reduced basis.

    One kernel replay substantiates authenticity: equality with the
    recomputed reduced Gröbner basis subsumes membership of every
    retained element in the source ideal.
    """

    if not generators_reduce_to_zero(ideal, groebner_basis, monomial_order):
        raise ValueError("retained basis does not reduce every ideal generator to zero")
    if not retained_basis_is_groebner(ideal, groebner_basis, monomial_order):
        raise ValueError("retained basis is not the Gröbner basis of the source ideal")


def _require_substantiated_budget_outcome(
    ideal: RationalPolynomialIdeal,
    polynomial: RationalPolynomial,
    groebner_basis: tuple[RationalPolynomial, ...] | None,
    monomial_order: str,
) -> None:
    """Require kernel evidence for a ``BUDGET_EXCEEDED`` outcome.

    With a retained basis, that basis must be authentic and its replayed
    reduction must genuinely overflow.  A stripped result is accepted only
    when the recomputed source Gröbner basis itself leaves the output
    budget.
    """

    if groebner_basis is None:
        if not retained_source_basis_exceeds_budget(ideal, monomial_order):
            raise ValueError(
                "BUDGET_EXCEEDED without a retained basis requires the "
                "source basis to exceed the output budget"
            )
        return
    _require_authentic_retained_basis(ideal, groebner_basis, monomial_order)
    if not replayed_remainder_exceeds_budget(
        ideal,
        groebner_basis,
        polynomial,
        monomial_order,
    ):
        raise ValueError("BUDGET_EXCEEDED contradicts an in-budget replayed reduction")


class IdealNormalFormResult(StrictModel):
    """The canonical remainder of one polynomial modulo an ideal.

    The result retains its complete source and the computed Gröbner basis so
    validation replays the defining reduction relation exactly.  When the
    exact remainder exceeds the 1,024-term output boundary the status reports
    ``BUDGET_EXCEEDED`` instead of a mathematical conclusion; no partial
    remainder is returned.
    """

    ideal: RationalPolynomialIdeal
    polynomial: RationalPolynomial
    monomial_order: Literal["lex", "grlex", "grevlex"]
    status: Literal["COMPUTED", "BUDGET_EXCEEDED"]
    groebner_basis: tuple[RationalPolynomial, ...] | None
    remainder: RationalPolynomial | None

    @model_validator(mode="after")
    def require_replayable_reduction(self) -> Self:
        _validate_membership_source(self.ideal, self.polynomial)
        _validate_retained_basis(self.groebner_basis, self.ideal.variables)
        if self.status == "BUDGET_EXCEEDED":
            if self.remainder is not None:
                raise ValueError("BUDGET_EXCEEDED must not carry a remainder")
            _require_substantiated_budget_outcome(
                self.ideal,
                self.polynomial,
                self.groebner_basis,
                self.monomial_order,
            )
            return self
        if self.groebner_basis is None or self.remainder is None:
            raise ValueError(
                "COMPUTED requires both the retained basis and the remainder"
            )
        if self.remainder.variables != self.ideal.variables:
            raise ValueError("remainder must use the same ring as the ideal")
        _require_authentic_retained_basis(
            self.ideal, self.groebner_basis, self.monomial_order
        )
        if not remainder_matches_claim(
            self.ideal,
            self.groebner_basis,
            self.polynomial,
            self.monomial_order,
            self.remainder,
        ):
            raise ValueError(
                "remainder does not replay against the retained Gröbner basis"
            )
        return self


class IdealMembershipResult(StrictModel):
    """Whether a polynomial lies in an ideal, with the normal form.

    ``IN_IDEAL`` is defined exactly as the normal form being zero and
    ``NOT_IN_IDEAL`` as it being nonzero; both carry the full defining
    relation (source ideal, source polynomial, monomial order, computed
    Gröbner basis) so validation can replay the reduction.  When the exact
    normal form exceeds the output boundary the status reports
    ``BUDGET_EXCEEDED`` and asserts no membership conclusion.
    """

    ideal: RationalPolynomialIdeal
    polynomial: RationalPolynomial
    monomial_order: Literal["lex", "grlex", "grevlex"]
    status: Literal["IN_IDEAL", "NOT_IN_IDEAL", "BUDGET_EXCEEDED"]
    groebner_basis: tuple[RationalPolynomial, ...] | None
    normal_form: RationalPolynomial | None

    @model_validator(mode="after")
    def require_replayable_membership(self) -> Self:
        _validate_membership_source(self.ideal, self.polynomial)
        _validate_retained_basis(self.groebner_basis, self.ideal.variables)
        if self.status == "BUDGET_EXCEEDED":
            if self.normal_form is not None:
                raise ValueError("BUDGET_EXCEEDED must not carry a normal form")
            _require_substantiated_budget_outcome(
                self.ideal,
                self.polynomial,
                self.groebner_basis,
                self.monomial_order,
            )
            return self
        if self.groebner_basis is None or self.normal_form is None:
            raise ValueError(f"{self.status} requires the basis and the normal form")
        if self.normal_form.variables != self.ideal.variables:
            raise ValueError("normal form must use the same ring as the ideal")
        normal_form_is_zero = not self.normal_form.polynomial.terms
        if (self.status == "IN_IDEAL") is not normal_form_is_zero:
            raise ValueError("status must equal (normal_form == 0)")
        _require_authentic_retained_basis(
            self.ideal, self.groebner_basis, self.monomial_order
        )
        if not remainder_matches_claim(
            self.ideal,
            self.groebner_basis,
            self.polynomial,
            self.monomial_order,
            self.normal_form,
        ):
            raise ValueError(
                "normal form does not replay against the retained Gröbner basis"
            )
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
    "IdealMembershipRequest",
    "IdealMembershipResult",
    "IdealNormalFormResult",
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
