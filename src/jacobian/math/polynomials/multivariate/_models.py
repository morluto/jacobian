"""Contracts for exact multivariate polynomial operations over ``QQ``."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import reduce
from math import comb, factorial, gcd, lcm
from typing import Any, Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.polynomials.values import (
    PolynomialVariable,
    RationalPolynomial,
    require_polynomial_budget,
)


def _validation_error(message: str) -> PydanticCustomError:
    return PydanticCustomError("polynomial.multivariate_contract", message)


_MAX_MULTIVARIATE_TERMS = 512
_MAX_MULTIVARIATE_EXPONENT = 64
_MAX_MULTIVARIATE_COEFFICIENT_DIGITS = 256
_MAX_ELIMINATION_DEGREE_SUM = 64
# Public output-term budget for one converted irreducible factor.  The
# operation converter uses this same bound; keeping it here lets the result
# validator reproduce the kernel's exact exceedance decision.
_MAX_FACTOR_OUTPUT_TERMS = 1_024

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

_MULTIVARIATE_MIN_VARIABLES = 2


def _validate_multivariate_pair(
    left: RationalPolynomial,
    right: RationalPolynomial,
) -> None:
    """Shared validation for two polynomials in the same declared ring."""

    if len(left.variables) < _MULTIVARIATE_MIN_VARIABLES:
        raise _validation_error(
            "multivariate operations require at least two variables"
        )
    if left.variables != right.variables:
        raise _validation_error("both polynomials must use the same ordered variables")


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

    # The Brown kernel also materializes coefficient-ring scaling powers:
    # every pseudo-remainder ends by multiplying with its divisor's leading
    # coefficient raised to the degree gap plus one, and the abnormal branch
    # raises scalar subresultants and member leading coefficients to the gap.
    # A power's support follows from its base's remaining-variable degree
    # times the exponent, so the formal returned supports above do not bound
    # it.  The first gap is higher_degree - lower_degree and every later gap
    # is at most lower_degree; every power base (a source or member leading
    # coefficient, or a scalar subresultant) has remaining degree at most the
    # index-zero formal coefficient degree.  Pseudo-remainder running
    # coefficients additionally retain exactly one source coefficient while
    # appending one divisor-side factor per elimination step -- at most
    # ``power_exponent`` of them -- so the power allowance carries one extra
    # highest source remaining degree.  Folding this derived power support
    # into ``maximum_support`` keeps the product invariant below true
    # for the scaling factors as well, and rejects the abnormal-gap nonscalar
    # regime whose powers would otherwise expand unboundedly.
    power_base_remaining_degree = (
        lower_degree * higher_remaining_degree + higher_degree * lower_remaining_degree
    )
    power_exponent = max(higher_degree - lower_degree, lower_degree) + 1
    scaling_power_support = comb(
        power_exponent * power_base_remaining_degree
        + max(higher_remaining_degree, lower_remaining_degree)
        + remaining_variable_count,
        remaining_variable_count,
    )
    maximum_support = max(maximum_support, scaling_power_support)

    sylvester_order = higher_degree + lower_degree
    # At most ``sylvester_order`` PRS steps each perform at most a quadratic
    # number of coefficient-ring operations.  Every materialized quantity --
    # sources, members, scalar subresultants, and their scaling-power
    # products -- has support at most ``maximum_support``, so every
    # coefficient product expands at most ``maximum_support**2`` term pairs.
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
            raise _validation_error("main variable must belong to the declared ring")
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
            raise _validation_error(
                "both polynomials must have positive main-variable degree"
            )
        if sum(degrees) > _MAX_ELIMINATION_DEGREE_SUM:
            raise _validation_error(
                "Sylvester order exceeds the subresultant backend budget"
            )

        envelope = _subresultant_envelope(self.left, self.right, variable_index)
        if envelope.aggregate_terms > _MAX_SUBRESULTANT_SEQUENCE_TERMS:
            raise _validation_error(
                "formal subresultant sequence support exceeds the aggregate "
                "result-term budget"
            )
        if envelope.maximum_coefficient_support > _MAX_SUBRESULTANT_COEFFICIENT_SUPPORT:
            raise _validation_error(
                "subresultant coefficient support exceeds the intermediate "
                "polynomial-term budget"
            )
        if envelope.arithmetic_term_pairs > _MAX_SUBRESULTANT_ARITHMETIC_TERM_PAIRS:
            raise _validation_error(
                "subresultant pseudo-remainder arithmetic exceeds the term-pair "
                "work budget"
            )
        if envelope.coefficient_bits > _MAX_SUBRESULTANT_COEFFICIENT_BITS:
            raise _validation_error(
                "subresultant determinant coefficient height exceeds the exact "
                "coefficient-bit budget"
            )
        if (
            envelope.intermediate_coefficient_bits
            > _MAX_SUBRESULTANT_INTERMEDIATE_COEFFICIENT_BITS
        ):
            raise _validation_error(
                "Brown pseudo-remainder intermediate coefficient height exceeds "
                "the exact coefficient-bit budget"
            )
        if (
            envelope.serialized_coefficient_bits
            > _MAX_SUBRESULTANT_SERIALIZED_COEFFICIENT_BITS
        ):
            raise _validation_error(
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
            raise _validation_error("source_order does not match main-variable degrees")
        variable_index = request.left.variables.index(request.main_variable)
        left_degree = _degree_in_variable(request.left, variable_index)
        right_degree = _degree_in_variable(request.right, variable_index)
        expected_resultant_sign = (
            -1
            if expected_order == "RIGHT_LEFT" and left_degree * right_degree % 2 == 1
            else 1
        )
        if self.resultant_sign_from_sequence_order != expected_resultant_sign:
            raise _validation_error(
                "resultant sign does not match the PRS source order"
            )
        actual_polynomials = tuple(member.polynomial for member in self.members)
        if actual_polynomials != expected_polynomials:
            raise _validation_error(
                "members do not replay the exact subresultant sequence"
            )

        expected_degrees = tuple(
            _degree_in_variable(polynomial, variable_index)
            for polynomial in expected_polynomials
        )
        actual_degrees = tuple(
            member.degree_in_main_variable for member in self.members
        )
        if actual_degrees != expected_degrees:
            raise _validation_error("member degrees do not match their polynomials")
        greatest_degree = max(expected_degrees)
        expected_degree_set = set(expected_degrees)
        expected_absent = tuple(
            degree
            for degree in range(greatest_degree + 1)
            if degree not in expected_degree_set
        )
        if self.skipped_member_degrees != expected_absent:
            raise _validation_error(
                "skipped_member_degrees does not match the PRS ledger"
            )
        expected_scalar_ledger = tuple(
            MultivariatePrincipalSubresultantCoefficient(
                index=index,
                coefficient=coefficient,
            )
            for index, coefficient in enumerate(expected_principal_coefficients)
        )
        if self.principal_subresultant_coefficients != expected_scalar_ledger:
            raise _validation_error(
                "principal subresultant coefficients do not replay the source pair"
            )
        if self.gcd_member_index != len(self.members) - 1:
            raise _validation_error("gcd_member_index must select the final PRS member")

        expected_resultant = polynomial_resultant_in_remaining_ring(
            request.left,
            request.right,
            request.main_variable,
            maximum_terms=_MAX_SUBRESULTANT_SEQUENCE_TERMS,
        )
        if self.resultant != expected_resultant:
            raise _validation_error("resultant does not match the source pair")
        expected_gcd_degree, expected_leading_coefficient = fraction_field_gcd_relation(
            request.left,
            request.right,
            request.main_variable,
            self.members[self.gcd_member_index].polynomial,
            maximum_terms=_MAX_SUBRESULTANT_SEQUENCE_TERMS,
        )
        if self.gcd_degree_in_main_variable != expected_gcd_degree:
            raise _validation_error("gcd degree does not match the final PRS member")
        if self.gcd_member_leading_coefficient != expected_leading_coefficient:
            raise _validation_error(
                "gcd leading coefficient does not match the final member"
            )
        return self


class MultivariateFactorRequest(StrictModel):
    """Exact factorization request over ``QQ[variables]`` for nonzero multivariate polynomials."""

    polynomial: RationalPolynomial = Field(
        description=(
            "Nonzero multivariate polynomial in QQ[variables] with at least "
            "two variables (univariate factorization is owned by "
            "polynomial.factor.compute); terms, exponents, and coefficients "
            "must respect the operation's exact budget."
        )
    )

    @model_validator(mode="after")
    def require_factor_budget(self) -> Self:
        if len(self.polynomial.variables) < _MULTIVARIATE_MIN_VARIABLES:
            raise _validation_error(
                f"multivariate factorization requires at least {_MULTIVARIATE_MIN_VARIABLES} variables; "
                "univariate polynomials are handled by polynomial.factor.compute"
            )
        if not self.polynomial.polynomial.terms:
            raise _validation_error("zero polynomial has no factorization")
        require_polynomial_budget(
            self.polynomial,
            maximum_terms=_MAX_MULTIVARIATE_TERMS,
            maximum_exponent=_MAX_MULTIVARIATE_EXPONENT,
            maximum_coefficient_digits=_MAX_MULTIVARIATE_COEFFICIENT_DIGITS,
        )
        _require_representable_content(self.polynomial)
        return self


def _require_representable_content(polynomial: RationalPolynomial) -> None:
    """Bound the aggregate rational content before any backend expansion.

    Clearing denominators to the least common multiple of all term
    denominators produces the primitive integer representative, so per-term
    digit budgets bound neither the rational content every result carries as
    one canonical rational nor the primitive coefficients the reconstructed
    polynomial and its monic factors must publish.  Wire terms are reduced,
    so the cleared content is already reduced, and both derived envelopes
    are checked here with exact integer arithmetic on the admitted terms.
    """

    fractions = [term.coefficient.as_fraction() for term in polynomial.polynomial.terms]
    common_denominator = reduce(lcm, (value.denominator for value in fractions), 1)
    scaled = [
        value.numerator * (common_denominator // value.denominator)
        for value in fractions
    ]
    content_numerator = gcd(*scaled)
    canonical_bound = 10**MAX_CANONICAL_RATIONAL_DIGITS
    if (
        abs(content_numerator) >= canonical_bound
        or common_denominator >= canonical_bound
    ):
        raise _validation_error(
            "aggregate rational content exceeds the "
            f"{MAX_CANONICAL_RATIONAL_DIGITS}-digit representable bound"
        )
    primitive_bound = 10**_MAX_MULTIVARIATE_COEFFICIENT_DIGITS
    for value in scaled:
        if abs(value // content_numerator) >= primitive_bound:
            raise _validation_error(
                "primitive integer coefficients exceed the "
                f"{_MAX_MULTIVARIATE_COEFFICIENT_DIGITS}-digit operation budget"
            )


class MultivariateIrreducibleFactor(StrictModel):
    factor: RationalPolynomial
    multiplicity: int = Field(ge=1, le=_MAX_MULTIVARIATE_EXPONENT)


class MultivariateFactorResult(StrictModel):
    """Exact factorization outcome over ``QQ[variables]``.

    ``FACTORIZED`` carries the full content-and-monic-irreducibles
    decomposition.  ``OUTPUT_BUDGET_EXCEEDED`` reports, as a typed bounded
    outcome, that the exact factorization is beyond this operation's
    public output bounds: either an irreducible factor exceeds the public
    output-term budget or the serialized exact decomposition exceeded the
    declared transport bound.  ``EXECUTION_FAILED`` is not a mathematical
    conclusion: the worker was stopped by its deadline or cancellation,
    killed by an enforced resource cap such as its CPU or address-space
    budget, crashed, or could not be contained, so no factorization was
    obtained and callers may retry.
    For both non-FACTORIZED statuses ``reconstructed`` restates the
    requested polynomial unchanged, ``coefficient`` carries the exact
    positive rational content of that polynomial, and ``factors`` is empty.
    """

    status: Literal[
        "FACTORIZED",
        "OUTPUT_BUDGET_EXCEEDED",
        "EXECUTION_FAILED",
    ] = "FACTORIZED"
    coefficient: CanonicalRational
    factors: tuple[MultivariateIrreducibleFactor, ...] = Field(max_length=128)
    reconstructed: RationalPolynomial
    normalization: Literal["CONTENT_AND_MONIC_IRREDUCIBLES"] | None = (
        "CONTENT_AND_MONIC_IRREDUCIBLES"
    )
    product_reconstruction: Literal["EXACT"] | None = "EXACT"

    @model_validator(mode="after")
    def require_canonical(self) -> Self:
        if self.coefficient.as_fraction() == 0:
            raise _validation_error("factorization coefficient must be nonzero")
        require_polynomial_budget(
            self.reconstructed,
            maximum_terms=_MAX_MULTIVARIATE_TERMS,
            maximum_exponent=_MAX_MULTIVARIATE_EXPONENT,
            maximum_coefficient_digits=_MAX_MULTIVARIATE_COEFFICIENT_DIGITS,
        )
        if self.status != "FACTORIZED":
            from jacobian.math.polynomials.multivariate import _factor_backend

            if self.factors:
                raise _validation_error(
                    "non-FACTORIZED outcomes carry no irreducible factors"
                )
            if (
                self.normalization is not None
                or self.product_reconstruction is not None
            ):
                raise _validation_error(
                    "non-FACTORIZED outcomes declare no normalization or "
                    "product reconstruction"
                )
            if _factor_backend.primitive_content_fraction(self.reconstructed) != (
                self.coefficient.as_fraction()
            ):
                raise _validation_error(
                    "outcome coefficient does not match the exact content "
                    "of the restated polynomial"
                )
            if self.status == "OUTPUT_BUDGET_EXCEEDED":
                # A capacity claim IS a mathematical claim about the exact
                # output, so replay it.  An interrupted replay establishes
                # nothing and must not authenticate the claim; a replayed
                # resource/output exceedance does.
                _verify_output_budget_exceeded_claim(
                    self.coefficient, self.reconstructed
                )
            return self
        if (
            self.normalization != "CONTENT_AND_MONIC_IRREDUCIBLES"
            or self.product_reconstruction != "EXACT"
        ):
            raise _validation_error(
                "FACTORIZED outcomes declare content-and-monic-irreducibles "
                "normalization and exact product reconstruction"
            )
        if not self.reconstructed.polynomial.terms:
            raise _validation_error("reconstructed polynomial must be nonzero")
        _check_factor_records(self.factors, self.reconstructed.variables)
        _require_aggregate_degree_consistent(self.factors, self.reconstructed)
        _require_distinct_canonical_order(self.factors)
        _verify_monic_irreducibles(self.factors)
        _verify_exact_reconstruction(
            self.coefficient,
            self.factors,
            self.reconstructed,
        )
        return self


_FactorContentKey = tuple[tuple[tuple[int, ...], str, str], ...]
_SympyFactorKey = tuple[tuple[tuple[int, ...], int, int], ...]


def _factor_content_key(record: MultivariateIrreducibleFactor) -> _FactorContentKey:
    return tuple(
        (term.exponents, term.coefficient.num, term.coefficient.den)
        for term in record.factor.polynomial.terms
    )


def _factor_total_degree(record: MultivariateIrreducibleFactor) -> int:
    return max(
        (sum(term.exponents) for term in record.factor.polynomial.terms),
        default=0,
    )


def _reconstructed_total_degree(reconstructed: RationalPolynomial) -> int:
    return max(
        (sum(term.exponents) for term in reconstructed.polynomial.terms),
        default=0,
    )


def _require_aggregate_degree_consistent(
    factors: tuple[MultivariateIrreducibleFactor, ...],
    reconstructed: RationalPolynomial,
) -> None:
    """Reject aggregate degree mismatches before any product expansion.

    The exact product's total degree equals the multiplicity-weighted sum of
    factor degrees, so a prefix overshoot proves the payload cannot satisfy
    the defining invariant without expanding a prohibitive dense product.
    """

    target = _reconstructed_total_degree(reconstructed)
    aggregate = 0
    for record in factors:
        aggregate += _factor_total_degree(record) * record.multiplicity
        if aggregate > target:
            raise _validation_error(
                "aggregate irreducible degree exceeds the reconstructed "
                "total degree; the factorization product cannot match"
            )


def _monic_content_fraction(content: Any) -> Fraction:
    """Extract the exact rational content returned by ``_monic_decomposition``."""

    leading = getattr(content, "LC", None)
    value = leading() if callable(leading) else content
    return Fraction(int(value.p), int(value.q))


def _verify_output_budget_exceeded_claim(
    coefficient: CanonicalRational,
    reconstructed: RationalPolynomial,
) -> None:
    """Re-derive a claimed ``OUTPUT_BUDGET_EXCEEDED`` status from its source.

    Replays the kernel's own bounded factorization so the reported
    incompleteness is bound to the restated polynomial instead of being an
    authorable label: the exact rational content is recomputed cheaply and
    compared, and the claim is reproduced only when a replayed factor
    conversion exceeds the output-term budget or the replayed run again
    exceeds the declared transport bound on the serialized decomposition.
    An interrupted replay — deadline, cancellation, or resource-cap kill
    such as worker memory exhaustion — establishes nothing about output
    size and fails closed instead of authenticating the claim.
    """

    from jacobian.math.polynomials.multivariate import _factor_backend
    from jacobian.math.polynomials.multivariate._factor_backend import (
        FactorBackendExhaustedError,
        FactorBackendInterruptedError,
    )

    if _factor_backend.primitive_content_fraction(reconstructed) != (
        coefficient.as_fraction()
    ):
        raise _validation_error(
            "budget-exceeded outcome coefficient does not match the exact "
            "content of the restated polynomial"
        )
    try:
        decomposition = _factor_backend.run_bounded_factorization(
            reconstructed,
            wall_seconds=_factor_backend.FACTOR_VERIFY_WALL_SECONDS,
        )
    except FactorBackendExhaustedError:
        # The replay hit the same declared transport bound on the
        # serialized decomposition: the claimed beyond-bounds behavior of
        # this exact source is reproduced.  An interrupted replay —
        # deadline, cancellation, or a resource-cap kill such as worker
        # memory exhaustion — proves nothing and must not validate the
        # claim, so only exhaustion returns here.
        return
    except FactorBackendInterruptedError as exc:
        raise _validation_error(
            "budget-exceeded outcome could not be re-derived because the "
            "verification replay was itself stopped before completing"
        ) from exc
    from jacobian.math.polynomials._conversions import (
        rational_polynomial_from_sympy,
        rational_polynomial_to_sympy,
    )
    from jacobian.math.polynomials._sympy import _monic_decomposition

    source = rational_polynomial_to_sympy(reconstructed)
    _content, raw_factors, _reconstructed = _monic_decomposition(
        source,
        decomposition,
        label="multivariate factorization",
    )
    for factor, _multiplicity in raw_factors:
        try:
            rational_polynomial_from_sympy(
                factor,
                reconstructed.variables,
                maximum_terms=_MAX_FACTOR_OUTPUT_TERMS,
            )
        except ValueError as exc:
            if "term operation budget" in str(exc):
                return
            raise
    raise _validation_error(
        "claimed output-budget exceedance is not reproduced by the exact "
        "factorization of the restated polynomial"
    )


def _check_factor_records(
    factors: tuple[MultivariateIrreducibleFactor, ...],
    variables: tuple[str, ...],
) -> None:
    """Enforce the reconstruction-safe envelope before any SymPy expansion.

    Factor records are kernel outputs, so their term budget is the output
    conversion budget rather than the request envelope.
    """

    for record in factors:
        if record.factor.variables != variables:
            raise _validation_error("irreducible factors must use the source ring")
        require_polynomial_budget(
            record.factor,
            maximum_terms=_MAX_FACTOR_OUTPUT_TERMS,
            maximum_exponent=_MAX_MULTIVARIATE_EXPONENT,
            maximum_coefficient_digits=_MAX_MULTIVARIATE_COEFFICIENT_DIGITS,
        )
        if _factor_total_degree(record) == 0:
            raise _validation_error("irreducible factor must be non-constant")


def _require_distinct_canonical_order(
    factors: tuple[MultivariateIrreducibleFactor, ...],
) -> None:
    seen: set[_FactorContentKey] = set()
    for key in (_factor_content_key(record) for record in factors):
        if key in seen:
            raise _validation_error("irreducible factors must be distinct")
        seen.add(key)
    ordered = tuple(
        sorted(
            factors,
            key=lambda record: (
                record.multiplicity,
                _factor_total_degree(record),
                _factor_content_key(record),
            ),
        ),
    )
    if factors != ordered:
        raise _validation_error("irreducible factors must use canonical order")


def _require_monic(poly: Any, factor: RationalPolynomial) -> None:
    lc = poly.LC()
    if getattr(lc, "p", None) != 1 or getattr(lc, "q", None) != 1:
        raise _validation_error(f"irreducible factor {factor} is not monic")


def _verify_monic_irreducibles(
    factors: tuple[MultivariateIrreducibleFactor, ...],
) -> None:
    """Enforce CONTENT_AND_MONIC_IRREDUCIBLES on every listed factor."""

    from jacobian.math.polynomials._conversions import rational_polynomial_to_sympy

    for record in factors:
        poly = rational_polynomial_to_sympy(record.factor)
        try:
            _require_monic(poly, record.factor)
            if not poly.is_irreducible:
                raise _validation_error(f"factor {record.factor} is not irreducible")
        except ValueError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise _validation_error("invalid factor normalization check") from exc


def _sympy_factor_key(poly: Any) -> _SympyFactorKey:
    """Return the canonical hashable form of one monic QQ ``Poly``."""

    return tuple(
        sorted(
            (tuple(monom), int(coeff.p), int(coeff.q)) for monom, coeff in poly.terms()
        )
    )


def _verify_exact_reconstruction(
    coefficient: CanonicalRational,
    factors: tuple[MultivariateIrreducibleFactor, ...],
    reconstructed: RationalPolynomial,
) -> None:
    """Check coefficient * ∏ factor**multiplicity == reconstructed exactly.

    The replay recomputes the exact content and the canonical monic
    irreducible multiset of the retained source polynomial with the same
    bounded, killable ``factor_list`` invocation the operation itself
    performs, then compares it against the claimed decomposition.  Monic
    irreducible factorization over ``QQ[variables]`` is unique, so matching
    multisets establish the product identity without expanding any
    intermediate product; partial products of admitted factorizations can be
    exponentially denser than their source (paired cyclotomic sums reach
    4^7 * 2 = 32,768 terms for an 8-variable input of 256 terms), so a
    division replay cannot carry a cofactor bound that covers every
    admitted factorization.  The verification cost is one killable worker
    call on the already-admitted source envelope.
    """

    from jacobian.math.polynomials._conversions import rational_polynomial_to_sympy
    from jacobian.math.polynomials._sympy import _monic_decomposition
    from jacobian.math.polynomials.multivariate import _factor_backend
    from jacobian.math.polynomials.multivariate._factor_backend import (
        FactorBackendExhaustedError,
        FactorBackendInterruptedError,
    )

    try:
        decomposition = _factor_backend.run_bounded_factorization(
            reconstructed,
            wall_seconds=_factor_backend.FACTOR_VERIFY_WALL_SECONDS,
        )
        source = rational_polynomial_to_sympy(reconstructed)
        content, raw_factors, _ = _monic_decomposition(
            source,
            decomposition,
            label="multivariate factorization",
        )
        claimed: dict[_SympyFactorKey, int] = {}
        for record in factors:
            key = _sympy_factor_key(rational_polynomial_to_sympy(record.factor))
            claimed[key] = claimed.get(key, 0) + record.multiplicity
        replayed: dict[_SympyFactorKey, int] = {}
        for factor, multiplicity in raw_factors:
            key = _sympy_factor_key(factor)
            replayed[key] = replayed.get(key, 0) + multiplicity
        if (
            _monic_content_fraction(content) != coefficient.as_fraction()
            or claimed != replayed
        ):
            raise _validation_error(
                "factorization product does not equal reconstructed polynomial"
            )
    except ValueError:
        raise
    except (FactorBackendExhaustedError, FactorBackendInterruptedError) as exc:
        raise _validation_error(
            "factorization verification could not reproduce the exact "
            "factorization within the declared work budget"
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise _validation_error("invalid factorization reconstruction") from exc


__all__ = [
    "MultivariateFactorRequest",
    "MultivariateFactorResult",
    "MultivariateIrreducibleFactor",
    "MultivariatePrincipalSubresultantCoefficient",
    "MultivariateSubresultantMember",
    "MultivariateSubresultantSequenceRequest",
    "MultivariateSubresultantSequenceResult",
]
