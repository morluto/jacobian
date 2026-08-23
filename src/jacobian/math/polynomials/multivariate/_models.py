"""Contracts for exact multivariate polynomial operations over ``QQ``."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, factorial, lcm
from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.polynomials.values import (
    PolynomialVariable,
    RationalPolynomial,
    require_polynomial_budget,
)

_MAX_MULTIVARIATE_TERMS = 512
_MAX_MULTIVARIATE_EXPONENT = 64
_MAX_MULTIVARIATE_COEFFICIENT_DIGITS = 256
_MAX_ELIMINATION_DEGREE_SUM = 64
# The result converter rejects sparse outputs above this size.  The request
# validator uses the same bound to reject large possible supports before
# SymPy expands the Sylvester determinant.
_MAX_RESULTANT_TERMS = 1_024

# Subresultant admission follows the actual coefficient-ring expansion.  The
# full formal sequence support is bounded before SymPy enters its pseudo-
# remainder algorithm.  The arithmetic proxy includes production plus typed
# result replay (sequence and resultant twice, fraction-field GCD once).
_MAX_SUBRESULTANT_SEQUENCE_TERMS = 4_096
_MAX_SUBRESULTANT_COEFFICIENT_SUPPORT = 1_024
_MAX_SUBRESULTANT_ARITHMETIC_TERM_PAIRS = 8_000_000
_MAX_SUBRESULTANT_COEFFICIENT_BITS = 8_192
_MAX_SUBRESULTANT_INTERMEDIATE_COEFFICIENT_BITS = 8_192
_MAX_SUBRESULTANT_SERIALIZED_COEFFICIENT_BITS = 8_388_608
_SUBRESULTANT_BACKEND_PASS_COUNT = 5
_MAX_SUBRESULTANT_MEMBER_COUNT = _MAX_ELIMINATION_DEGREE_SUM // 2 + 2
_SUBRESULTANT_RESULT_TERM_MULTIPLIER = 4

MonomialOrder = Literal["lex", "grlex", "grevlex"]
"""Declared monomial order for multivariate polynomial division."""

_MULTIVARIATE_MIN_VARIABLES = 2


def _validate_multivariate_pair(
    left: RationalPolynomial,
    right: RationalPolynomial,
) -> None:
    """Shared validation for two polynomials in the same declared ring."""

    if len(left.variables) < _MULTIVARIATE_MIN_VARIABLES:
        raise ValueError("multivariate operations require at least two variables")
    if left.variables != right.variables:
        raise ValueError("both polynomials must use the same ordered variables")


def _resultant_support_bound(
    left: RationalPolynomial,
    right: RationalPolynomial,
    elimination_index: int,
) -> int:
    """Bound the number of possible monomials in a multivariate resultant.

    If ``f`` and ``g`` have elimination degrees ``m`` and ``n``, and total
    remaining-variable degrees ``d_f`` and ``d_g``, every resultant monomial
    has total degree at most ``n*d_f + m*d_g``.  The returned binomial is the
    number of monomials up to that degree in the remaining variables.
    """

    remaining_variable_count = len(left.variables) - 1
    if remaining_variable_count == 0:
        return 1

    def degree(polynomial: RationalPolynomial, *, in_remaining: bool) -> int:
        return max(
            (
                sum(
                    exponent
                    for index, exponent in enumerate(term.exponents)
                    if (index != elimination_index) == in_remaining
                )
                for term in polynomial.polynomial.terms
            ),
            default=0,
        )

    left_elimination_degree = degree(left, in_remaining=False)
    right_elimination_degree = degree(right, in_remaining=False)
    left_remaining_degree = degree(left, in_remaining=True)
    right_remaining_degree = degree(right, in_remaining=True)
    resultant_degree_bound = (
        right_elimination_degree * left_remaining_degree
        + left_elimination_degree * right_remaining_degree
    )
    return comb(
        resultant_degree_bound + remaining_variable_count,
        remaining_variable_count,
    )


def _degree_in_variable(polynomial: RationalPolynomial, variable_index: int) -> int:
    return max(
        (term.exponents[variable_index] for term in polynomial.polynomial.terms),
        default=0,
    )


def _remaining_total_degree(
    polynomial: RationalPolynomial,
    variable_index: int,
) -> int:
    return max(
        (
            sum(
                exponent
                for index, exponent in enumerate(term.exponents)
                if index != variable_index
            )
            for term in polynomial.polynomial.terms
        ),
        default=0,
    )


def _maximum_coefficient_support(
    polynomial: RationalPolynomial,
    variable_index: int,
) -> int:
    support_by_degree: dict[int, int] = {}
    for term in polynomial.polynomial.terms:
        degree = term.exponents[variable_index]
        support_by_degree[degree] = support_by_degree.get(degree, 0) + 1
    return max(support_by_degree.values(), default=0)


@dataclass(frozen=True)
class _SubresultantEnvelope:
    aggregate_terms: int
    maximum_coefficient_support: int
    arithmetic_term_pairs: int
    coefficient_bits: int
    intermediate_coefficient_bits: int
    serialized_coefficient_bits: int


def _subresultant_coefficient_height_bits(
    left: RationalPolynomial,
    right: RationalPolynomial,
    *,
    sylvester_order: int,
    input_coefficient_support: int,
) -> int:
    """Bound determinant coefficient numerators and denominators in bits.

    Put every input coefficient over their exact common denominator ``L``.
    A Sylvester minor has order at most ``sylvester_order``; Leibniz expansion
    then bounds a coefficient numerator by
    ``s! * (input_coefficient_support * scaled_height)^s`` and its denominator
    by ``L^s``.  Subresultant minors are no larger than the full Sylvester
    matrix, so the same bound covers every returned member.
    """

    coefficients = tuple(
        term.coefficient
        for polynomial in (left, right)
        for term in polynomial.polynomial.terms
    )
    common_denominator = 1
    ratios: list[tuple[int, int]] = []
    for coefficient in coefficients:
        numerator, denominator = coefficient.as_integer_ratio()
        ratios.append((numerator, denominator))
        common_denominator = lcm(common_denominator, denominator)
    scaled_height = max(
        (
            abs(numerator) * (common_denominator // denominator)
            for numerator, denominator in ratios
        ),
        default=1,
    )
    product_height = max(1, input_coefficient_support * scaled_height)
    numerator_bits = (
        factorial(sylvester_order).bit_length()
        + sylvester_order * product_height.bit_length()
    )
    denominator_bits = sylvester_order * max(1, common_denominator.bit_length())
    return max(numerator_bits, denominator_bits)


def _brown_intermediate_coefficient_height_bits(
    *,
    returned_coefficient_bits: int,
    sylvester_order: int,
    maximum_coefficient_support: int,
) -> int:
    """Bound coefficients materialized inside Brown pseudo-remainders.

    Every returned PRS coefficient and scalar subresultant is a Sylvester
    minor covered by ``returned_coefficient_bits``.  In SymPy's Brown kernel,
    one pseudo-remainder, scaling factor, exact-division partial product, or
    fraction-field GCD/replay intermediate is a sum of coefficient-ring
    products containing at most ``sylvester_order + 2`` such minor factors.

    All source coefficients share the exact common denominator used by the
    determinant bound.  Products therefore share a power of that denominator;
    summing convolution paths adds their logarithm rather than multiplying
    unrelated denominators.  ``maximum_coefficient_support`` bounds the paths
    contributed by each factor and the partial quotient ledger.  This is
    deliberately conservative, but it covers the pre-division pseudo-
    remainder that can be larger than every returned subresultant.
    """

    factor_count = sylvester_order + 2
    support_path_bits = (factor_count + 1) * max(
        1, maximum_coefficient_support.bit_length()
    )
    return (
        factor_count * returned_coefficient_bits
        + support_path_bits
        + factor_count.bit_length()
        + 2
    )


def _subresultant_envelope(
    left: RationalPolynomial,
    right: RationalPolynomial,
    variable_index: int,
) -> _SubresultantEnvelope:
    left_degree = _degree_in_variable(left, variable_index)
    right_degree = _degree_in_variable(right, variable_index)
    if left_degree >= right_degree:
        higher, lower = left, right
        higher_degree, lower_degree = left_degree, right_degree
    else:
        higher, lower = right, left
        higher_degree, lower_degree = right_degree, left_degree

    remaining_variable_count = len(left.variables) - 1
    higher_remaining_degree = _remaining_total_degree(higher, variable_index)
    lower_remaining_degree = _remaining_total_degree(lower, variable_index)
    aggregate_terms = len(left.polynomial.terms) + len(right.polynomial.terms)
    input_coefficient_support = max(
        _maximum_coefficient_support(left, variable_index),
        _maximum_coefficient_support(right, variable_index),
        1,
    )
    maximum_support = input_coefficient_support
    # The j-th formal subresultant has main-variable degree at most j.  Each
    # coefficient has remaining-variable total degree at most
    # (lower_degree-j)*deg_rest(higher) +
    # (higher_degree-j)*deg_rest(lower).
    for index in range(lower_degree + 1):
        coefficient_degree = (lower_degree - index) * higher_remaining_degree + (
            higher_degree - index
        ) * lower_remaining_degree
        coefficient_support = comb(
            coefficient_degree + remaining_variable_count,
            remaining_variable_count,
        )
        maximum_support = max(maximum_support, coefficient_support)
        aggregate_terms += (index + 1) * coefficient_support

    sylvester_order = higher_degree + lower_degree
    # At most ``sylvester_order`` PRS steps each perform at most a quadratic
    # number of coefficient-ring operations.  Every coefficient product
    # expands at most ``maximum_support**2`` term pairs.
    arithmetic_term_pairs = (
        _SUBRESULTANT_BACKEND_PASS_COUNT * sylvester_order**3 * maximum_support**2
    )
    coefficient_bits = _subresultant_coefficient_height_bits(
        left,
        right,
        sylvester_order=sylvester_order,
        input_coefficient_support=input_coefficient_support,
    )
    intermediate_coefficient_bits = _brown_intermediate_coefficient_height_bits(
        returned_coefficient_bits=coefficient_bits,
        sylvester_order=sylvester_order,
        maximum_coefficient_support=maximum_support,
    )
    return _SubresultantEnvelope(
        aggregate_terms=aggregate_terms,
        maximum_coefficient_support=maximum_support,
        arithmetic_term_pairs=arithmetic_term_pairs,
        coefficient_bits=coefficient_bits,
        intermediate_coefficient_bits=intermediate_coefficient_bits,
        # The result repeats the sources, returns the PRS and complete scalar-
        # subresultant ledger, and also carries the resultant and final-member
        # leading coefficient.  Four sequence-sized budgets cover them.
        serialized_coefficient_bits=(
            _SUBRESULTANT_RESULT_TERM_MULTIPLIER * aggregate_terms * coefficient_bits
        ),
    )


class MultivariateGcdRequest(StrictModel):
    """Two multivariate polynomials in ``QQ[x_1, ..., x_n]``."""

    left: RationalPolynomial
    right: RationalPolynomial

    @model_validator(mode="after")
    def require_multivariate_ring(self) -> Self:
        _validate_multivariate_pair(self.left, self.right)
        for polynomial in (self.left, self.right):
            require_polynomial_budget(
                polynomial,
                maximum_terms=_MAX_MULTIVARIATE_TERMS,
                maximum_exponent=_MAX_MULTIVARIATE_EXPONENT,
                maximum_coefficient_digits=_MAX_MULTIVARIATE_COEFFICIENT_DIGITS,
            )
        return self


class MultivariateGcdResult(StrictModel):
    gcd: RationalPolynomial
    convention: Literal["MONIC_ASSOCIATE"] = "MONIC_ASSOCIATE"


class MultivariateDivisionRequest(StrictModel):
    """Divide one multivariate polynomial by another under a declared monomial order."""

    left: RationalPolynomial
    right: RationalPolynomial
    monomial_order: MonomialOrder = "lex"

    @model_validator(mode="after")
    def require_multivariate_ring(self) -> Self:
        _validate_multivariate_pair(self.left, self.right)
        if not self.right.polynomial.terms:
            raise ValueError("divisor polynomial must be nonzero")
        for polynomial in (self.left, self.right):
            require_polynomial_budget(
                polynomial,
                maximum_terms=_MAX_MULTIVARIATE_TERMS,
                maximum_exponent=_MAX_MULTIVARIATE_EXPONENT,
                maximum_coefficient_digits=_MAX_MULTIVARIATE_COEFFICIENT_DIGITS,
            )
        return self


class MultivariateDivisionResult(StrictModel):
    quotient: RationalPolynomial
    remainder: RationalPolynomial
    monomial_order: MonomialOrder
    convention: Literal["EXACT_DIVISION_REMAINDER"] = "EXACT_DIVISION_REMAINDER"


class MultivariateResultantRequest(StrictModel):
    """Compute a bounded resultant with respect to one variable.

    The request rejects inputs whose degree envelope can produce more terms
    than the exact sparse result contract can represent.
    """

    left: RationalPolynomial
    right: RationalPolynomial
    elimination_variable: PolynomialVariable = Field(
        description="Variable eliminated by the Sylvester resultant.",
    )

    @model_validator(mode="after")
    def require_multivariate_ring(self) -> Self:
        _validate_multivariate_pair(self.left, self.right)
        if self.elimination_variable not in self.left.variables:
            raise ValueError("elimination variable must belong to the declared ring")
        for polynomial in (self.left, self.right):
            require_polynomial_budget(
                polynomial,
                maximum_terms=_MAX_MULTIVARIATE_TERMS,
                maximum_exponent=_MAX_MULTIVARIATE_EXPONENT,
                maximum_coefficient_digits=_MAX_MULTIVARIATE_COEFFICIENT_DIGITS,
            )
        variable_index = self.left.variables.index(self.elimination_variable)
        for polynomial, label in ((self.left, "left"), (self.right, "right")):
            degree_in_variable = max(
                (
                    term.exponents[variable_index]
                    for term in polynomial.polynomial.terms
                ),
                default=0,
            )
            if degree_in_variable == 0:
                raise ValueError(
                    f"{label} polynomial has zero degree in the elimination variable"
                )
        degree_sum = max(
            (term.exponents[variable_index] for term in self.left.polynomial.terms),
            default=0,
        ) + max(
            (term.exponents[variable_index] for term in self.right.polynomial.terms),
            default=0,
        )
        if degree_sum > _MAX_ELIMINATION_DEGREE_SUM:
            raise ValueError("Sylvester degree exceeds the resultant budget")
        if (
            _resultant_support_bound(self.left, self.right, variable_index)
            > _MAX_RESULTANT_TERMS
        ):
            raise ValueError("resultant output exceeds the term budget")
        return self


class MultivariateScalarValue(StrictModel):
    kind: Literal["SCALAR"] = "SCALAR"
    value: CanonicalRational


class MultivariatePolynomialValue(StrictModel):
    kind: Literal["POLYNOMIAL"] = "POLYNOMIAL"
    value: RationalPolynomial


MultivariateInvariantValue = Annotated[
    MultivariateScalarValue | MultivariatePolynomialValue,
    Field(discriminator="kind"),
]


class MultivariateResultantResult(StrictModel):
    elimination_variable: PolynomialVariable
    resultant: MultivariateInvariantValue
    convention: Literal["SYLVESTER_DETERMINANT"] = "SYLVESTER_DETERMINANT"


class MultivariateSubresultantSequenceRequest(StrictModel):
    """Compute a bounded Brown subresultant PRS in one declared variable.

    Both source values belong to the same ordered ``QQ`` polynomial ring.  The
    main variable has positive degree in both inputs; every other declared
    variable remains an exact polynomial coefficient rather than being
    specialized or promoted to an ambient expression.
    """

    left: RationalPolynomial
    right: RationalPolynomial
    main_variable: PolynomialVariable = Field(
        description=(
            "Variable used for the univariate subresultant sequence; all "
            "other declared variables form its exact QQ polynomial "
            "coefficient ring."
        ),
        examples=["x"],
    )

    @model_validator(mode="after")
    def require_bounded_subresultant_envelope(self) -> Self:
        _validate_multivariate_pair(self.left, self.right)
        if self.main_variable not in self.left.variables:
            raise ValueError("main variable must belong to the declared ring")
        for polynomial, label in ((self.left, "left"), (self.right, "right")):
            require_polynomial_budget(
                polynomial,
                maximum_terms=_MAX_MULTIVARIATE_TERMS,
                maximum_exponent=_MAX_MULTIVARIATE_EXPONENT,
                maximum_coefficient_digits=_MAX_MULTIVARIATE_COEFFICIENT_DIGITS,
                label=f"{label} polynomial",
            )
        variable_index = self.left.variables.index(self.main_variable)
        degrees = (
            _degree_in_variable(self.left, variable_index),
            _degree_in_variable(self.right, variable_index),
        )
        if any(degree == 0 for degree in degrees):
            raise ValueError("both polynomials must have positive main-variable degree")
        if sum(degrees) > _MAX_ELIMINATION_DEGREE_SUM:
            raise ValueError("Sylvester order exceeds the subresultant backend budget")

        envelope = _subresultant_envelope(self.left, self.right, variable_index)
        if envelope.aggregate_terms > _MAX_SUBRESULTANT_SEQUENCE_TERMS:
            raise ValueError(
                "formal subresultant sequence support exceeds the aggregate "
                "result-term budget"
            )
        if envelope.maximum_coefficient_support > _MAX_SUBRESULTANT_COEFFICIENT_SUPPORT:
            raise ValueError(
                "subresultant coefficient support exceeds the intermediate "
                "polynomial-term budget"
            )
        if envelope.arithmetic_term_pairs > _MAX_SUBRESULTANT_ARITHMETIC_TERM_PAIRS:
            raise ValueError(
                "subresultant pseudo-remainder arithmetic exceeds the term-pair "
                "work budget"
            )
        if envelope.coefficient_bits > _MAX_SUBRESULTANT_COEFFICIENT_BITS:
            raise ValueError(
                "subresultant determinant coefficient height exceeds the exact "
                "coefficient-bit budget"
            )
        if (
            envelope.intermediate_coefficient_bits
            > _MAX_SUBRESULTANT_INTERMEDIATE_COEFFICIENT_BITS
        ):
            raise ValueError(
                "Brown pseudo-remainder intermediate coefficient height exceeds "
                "the exact coefficient-bit budget"
            )
        if (
            envelope.serialized_coefficient_bits
            > _MAX_SUBRESULTANT_SERIALIZED_COEFFICIENT_BITS
        ):
            raise ValueError(
                "subresultant sequence exceeds the aggregate exact-output budget"
            )
        return self


class MultivariateSubresultantMember(StrictModel):
    """One nonzero member of the source-bound subresultant PRS."""

    polynomial: RationalPolynomial
    degree_in_main_variable: StrictInt = Field(
        ge=0,
        le=_MAX_ELIMINATION_DEGREE_SUM,
    )


class MultivariatePrincipalSubresultantCoefficient(StrictModel):
    """One formal principal subresultant coefficient in ``QQ[rest]``."""

    index: StrictInt = Field(ge=0, le=_MAX_ELIMINATION_DEGREE_SUM // 2)
    coefficient: RationalPolynomial = Field(
        description=(
            "Coefficient of main_variable^index in the formal index-th "
            "subresultant for the PRS-ordered source pair. The canonical zero "
            "polynomial explicitly records a vanishing principal coefficient."
        )
    )


class MultivariateSubresultantSequenceResult(StrictModel):
    """Complete nonzero Brown PRS with exact degree and source binding.

    The maintained backend omits zero or same-degree PRS values.
    ``skipped_member_degrees`` records main-variable degrees without a distinct
    Brown member.  ``principal_subresultant_coefficients`` separately retains
    every formal scalar subresultant, including canonical zero polynomials, so
    a vanished coefficient is never conflated with a skipped PRS degree.
    Validation replays the pinned mathematical convention and independently
    checks the last member against the GCD over the remaining-variable fraction
    field.
    """

    left: RationalPolynomial
    right: RationalPolynomial
    main_variable: PolynomialVariable
    source_order: Literal["LEFT_RIGHT", "RIGHT_LEFT"]
    members: tuple[MultivariateSubresultantMember, ...] = Field(
        min_length=2,
        max_length=_MAX_SUBRESULTANT_MEMBER_COUNT,
        description=(
            "Nonzero Brown subresultant PRS members. The higher-degree "
            "source comes first; ties preserve left before right; subsequent "
            "members have strictly decreasing main-variable degree."
        ),
    )
    skipped_member_degrees: tuple[StrictInt, ...] = Field(
        max_length=_MAX_ELIMINATION_DEGREE_SUM,
        description=(
            "Increasing main-variable degrees from zero through the greatest "
            "source degree for which the nonzero Brown PRS has no distinct member."
        ),
    )
    principal_subresultant_coefficients: tuple[
        MultivariatePrincipalSubresultantCoefficient, ...
    ] = Field(
        min_length=1,
        max_length=_MAX_ELIMINATION_DEGREE_SUM // 2 + 1,
        description=(
            "Complete increasing ledger for formal indices zero through the "
            "lower PRS-ordered source degree. Coefficients live in "
            "the exact ordered ring of the remaining variables."
        ),
    )
    resultant: RationalPolynomial = Field(
        description=(
            "Sylvester resultant Res_main(left, right) in the ordered ring "
            "of the remaining variables, with the original left/right sign."
        ),
    )
    resultant_sign_from_sequence_order: Literal[-1, 1] = Field(
        description=(
            "Exact multiplier from the resultant of the PRS-ordered source "
            "pair to resultant: Res(left,right) = sign * Res(first,second)."
        )
    )
    gcd_member_index: StrictInt = Field(
        ge=1,
        le=_MAX_SUBRESULTANT_MEMBER_COUNT - 1,
        description=(
            "Index of the final PRS member, whose monic associate over the "
            "remaining-variable fraction field is gcd(left, right)."
        ),
    )
    gcd_degree_in_main_variable: StrictInt = Field(
        ge=0,
        le=_MAX_ELIMINATION_DEGREE_SUM,
    )
    gcd_member_leading_coefficient: RationalPolynomial = Field(
        description=(
            "Leading coefficient of the final PRS member in the remaining-"
            "variable polynomial ring. Dividing that member by this explicit "
            "fraction-field unit gives the monic gcd."
        )
    )
    convention: Literal["BROWN_SUBRESULTANT_PRS"] = "BROWN_SUBRESULTANT_PRS"
    zero_members_omitted: Literal[True] = True

    @model_validator(mode="after")
    def replay_source_bound_sequence(self) -> Self:
        request = MultivariateSubresultantSequenceRequest(
            left=self.left,
            right=self.right,
            main_variable=self.main_variable,
        )
        from jacobian.math.polynomials.multivariate._subresultants import (
            fraction_field_gcd_relation,
            polynomial_resultant_in_remaining_ring,
            polynomial_subresultant_sequence,
        )

        (
            expected_order,
            expected_polynomials,
            expected_principal_coefficients,
        ) = polynomial_subresultant_sequence(
            request.left,
            request.right,
            request.main_variable,
            maximum_terms=_MAX_SUBRESULTANT_SEQUENCE_TERMS,
        )
        if self.source_order != expected_order:
            raise ValueError("source_order does not match main-variable degrees")
        variable_index = request.left.variables.index(request.main_variable)
        left_degree = _degree_in_variable(request.left, variable_index)
        right_degree = _degree_in_variable(request.right, variable_index)
        expected_resultant_sign = (
            -1
            if expected_order == "RIGHT_LEFT" and left_degree * right_degree % 2 == 1
            else 1
        )
        if self.resultant_sign_from_sequence_order != expected_resultant_sign:
            raise ValueError("resultant sign does not match the PRS source order")
        actual_polynomials = tuple(member.polynomial for member in self.members)
        if actual_polynomials != expected_polynomials:
            raise ValueError("members do not replay the exact subresultant sequence")

        expected_degrees = tuple(
            _degree_in_variable(polynomial, variable_index)
            for polynomial in expected_polynomials
        )
        actual_degrees = tuple(
            member.degree_in_main_variable for member in self.members
        )
        if actual_degrees != expected_degrees:
            raise ValueError("member degrees do not match their polynomials")
        greatest_degree = max(expected_degrees)
        expected_degree_set = set(expected_degrees)
        expected_absent = tuple(
            degree
            for degree in range(greatest_degree + 1)
            if degree not in expected_degree_set
        )
        if self.skipped_member_degrees != expected_absent:
            raise ValueError("skipped_member_degrees does not match the PRS ledger")
        expected_scalar_ledger = tuple(
            MultivariatePrincipalSubresultantCoefficient(
                index=index,
                coefficient=coefficient,
            )
            for index, coefficient in enumerate(expected_principal_coefficients)
        )
        if self.principal_subresultant_coefficients != expected_scalar_ledger:
            raise ValueError(
                "principal subresultant coefficients do not replay the source pair"
            )
        if self.gcd_member_index != len(self.members) - 1:
            raise ValueError("gcd_member_index must select the final PRS member")

        expected_resultant = polynomial_resultant_in_remaining_ring(
            request.left,
            request.right,
            request.main_variable,
            maximum_terms=_MAX_SUBRESULTANT_SEQUENCE_TERMS,
        )
        if self.resultant != expected_resultant:
            raise ValueError("resultant does not match the source pair")
        expected_gcd_degree, expected_leading_coefficient = fraction_field_gcd_relation(
            request.left,
            request.right,
            request.main_variable,
            self.members[self.gcd_member_index].polynomial,
            maximum_terms=_MAX_SUBRESULTANT_SEQUENCE_TERMS,
        )
        if self.gcd_degree_in_main_variable != expected_gcd_degree:
            raise ValueError("gcd degree does not match the final PRS member")
        if self.gcd_member_leading_coefficient != expected_leading_coefficient:
            raise ValueError("gcd leading coefficient does not match the final member")
        return self


__all__ = [
    "MonomialOrder",
    "MultivariateDivisionRequest",
    "MultivariateDivisionResult",
    "MultivariateGcdRequest",
    "MultivariateGcdResult",
    "MultivariateInvariantValue",
    "MultivariatePolynomialValue",
    "MultivariatePrincipalSubresultantCoefficient",
    "MultivariateResultantRequest",
    "MultivariateResultantResult",
    "MultivariateScalarValue",
    "MultivariateSubresultantMember",
    "MultivariateSubresultantSequenceRequest",
    "MultivariateSubresultantSequenceResult",
]
