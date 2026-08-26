"""Private SymPy adapter for exact polynomial subresultant sequences."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, factorial, lcm
from typing import Any, Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._models import StrictModel
from jacobian.math.polynomials._conversions import (
    rational_polynomial_from_sympy,
    rational_polynomial_to_sympy,
    symbols_for_variables,
)
from jacobian.math.polynomials.multivariate._models import (
    _MAX_ELIMINATION_DEGREE_SUM,
    _MAX_MULTIVARIATE_COEFFICIENT_DIGITS,
    _MAX_MULTIVARIATE_EXPONENT,
    _MAX_MULTIVARIATE_TERMS,
    _degree_in_variable,
    _maximum_coefficient_support,
    _remaining_total_degree,
    _validate_multivariate_pair,
    _validation_error,
)
from jacobian.math.polynomials.values import (
    PolynomialVariable,
    RationalPolynomial,
    require_polynomial_budget,
)

SubresultantSourceOrder = Literal["LEFT_RIGHT", "RIGHT_LEFT"]


# Admission follows the actual coefficient-ring expansion. The complete formal
# sequence is bounded before SymPy enters its pseudo-remainder algorithm. The
# arithmetic proxy includes production and typed result replay.
_MAX_SUBRESULTANT_SEQUENCE_TERMS = 4_096
_MAX_SUBRESULTANT_COEFFICIENT_SUPPORT = 1_024
_MAX_SUBRESULTANT_ARITHMETIC_TERM_PAIRS = 8_000_000
_MAX_SUBRESULTANT_COEFFICIENT_BITS = 8_192
_MAX_SUBRESULTANT_INTERMEDIATE_COEFFICIENT_BITS = 8_192
_MAX_SUBRESULTANT_SERIALIZED_COEFFICIENT_BITS = 8_388_608
_SUBRESULTANT_BACKEND_PASS_COUNT = 5
_MAX_SUBRESULTANT_MEMBER_COUNT = _MAX_ELIMINATION_DEGREE_SUM // 2 + 2
_SUBRESULTANT_RESULT_TERM_MULTIPLIER = 4


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
    """Bound determinant coefficient numerators and denominators in bits."""

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
    """Bound coefficients materialized inside Brown pseudo-remainders."""

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
    source_order: SubresultantSourceOrder
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


def _as_univariate_over_polynomial_ring(
    polynomial: RationalPolynomial,
    main_variable: str,
) -> Any:
    """View one ``QQ[x_1, ..., x_n]`` value in ``QQ[rest][main]``."""

    from sympy import QQ, Poly

    variables = polynomial.variables
    symbols = symbols_for_variables(variables)
    main_index = variables.index(main_variable)
    remaining_symbols = tuple(
        symbol for index, symbol in enumerate(symbols) if index != main_index
    )
    coefficient_ring = QQ.poly_ring(*remaining_symbols)
    return Poly(
        rational_polynomial_to_sympy(polynomial).as_expr(),
        symbols[main_index],
        domain=coefficient_ring,
    )


def _as_full_ring_polynomial(
    polynomial: Any,
    variables: tuple[str, ...],
    *,
    maximum_terms: int,
) -> RationalPolynomial:
    from sympy import QQ, Poly

    full = Poly(
        polynomial.as_expr(),
        *symbols_for_variables(variables),
        domain=QQ,
    )
    return rational_polynomial_from_sympy(
        full,
        variables,
        maximum_terms=maximum_terms,
    )


def polynomial_subresultant_sequence(
    left: RationalPolynomial,
    right: RationalPolynomial,
    main_variable: str,
    *,
    maximum_terms: int,
) -> tuple[
    SubresultantSourceOrder,
    tuple[RationalPolynomial, ...],
    tuple[RationalPolynomial, ...],
]:
    """Return the Brown PRS and its complete principal-coefficient ledger.

    The pinned SymPy backend implements Brown's algorithm.  Jacobian fixes the
    public source-order convention here: the higher main-variable degree comes
    first and a tie preserves ``left, right``.  The third return value lists
    the scalar (principal) subresultant for every formal index from zero through
    the lower source degree.  A zero polynomial is retained, so a
    vanishing principal coefficient is distinct from a main-variable degree
    skipped by the nonzero PRS.
    """

    from sympy import Poly
    from sympy.polys.euclidtools import dup_inner_subresultants

    left_univariate = _as_univariate_over_polynomial_ring(left, main_variable)
    right_univariate = _as_univariate_over_polynomial_ring(right, main_variable)
    if left_univariate.degree() < right_univariate.degree():
        source_order: SubresultantSourceOrder = "RIGHT_LEFT"
        first, second = right_univariate, left_univariate
    else:
        source_order = "LEFT_RIGHT"
        first, second = left_univariate, right_univariate

    # ``Poly.subresultants`` delegates to this maintained Brown kernel but
    # discards its scalar-subresultant output.  Calling the same pinned kernel
    # here gives both values in one pass and lets the public result represent
    # zero principal coefficients explicitly.
    dense_sequence, nonzero_scalars = dup_inner_subresultants(
        first.rep.to_list(),
        second.rep.to_list(),
        first.domain,
    )
    sequence = tuple(
        Poly.from_list(
            dense_polynomial,
            gens=first.gens[0],
            domain=first.domain,
        )
        for dense_polynomial in dense_sequence
    )
    public_sequence = tuple(
        _as_full_ring_polynomial(
            polynomial,
            left.variables,
            maximum_terms=maximum_terms,
        )
        for polynomial in sequence
    )

    remaining_variables = tuple(
        variable for variable in left.variables if variable != main_variable
    )
    remaining_symbols = symbols_for_variables(remaining_variables)
    scalar_by_degree = {
        int(polynomial.degree()): scalar
        for polynomial, scalar in zip(sequence[1:], nonzero_scalars[1:], strict=True)
    }
    principal_coefficients = tuple(
        rational_polynomial_from_sympy(
            Poly(
                first.domain.to_sympy(scalar_by_degree.get(index, first.domain.zero)),
                *remaining_symbols,
                domain=first.domain.domain,
            ),
            remaining_variables,
            maximum_terms=maximum_terms,
        )
        for index in range(int(second.degree()) + 1)
    )
    return source_order, public_sequence, principal_coefficients


def polynomial_resultant_in_remaining_ring(
    left: RationalPolynomial,
    right: RationalPolynomial,
    main_variable: str,
    *,
    maximum_terms: int,
) -> RationalPolynomial:
    """Return ``Res_main(left, right)`` with the original argument orientation."""

    from sympy import QQ, Poly

    from jacobian.math.polynomials._sympy import polynomial_resultant

    variables = left.variables
    main_index = variables.index(main_variable)
    symbols = symbols_for_variables(variables)
    remaining_variables = tuple(
        variable for index, variable in enumerate(variables) if index != main_index
    )
    remaining_symbols = tuple(
        symbol for index, symbol in enumerate(symbols) if index != main_index
    )
    left_polynomial = rational_polynomial_to_sympy(left)
    right_polynomial = rational_polynomial_to_sympy(right)
    value = polynomial_resultant(
        left_polynomial,
        right_polynomial,
        symbols[main_index],
    )
    polynomial = Poly(value, *remaining_symbols, domain=QQ)
    return rational_polynomial_from_sympy(
        polynomial,
        remaining_variables,
        maximum_terms=maximum_terms,
    )


def fraction_field_gcd_relation(
    left: RationalPolynomial,
    right: RationalPolynomial,
    main_variable: str,
    gcd_member: RationalPolynomial,
    *,
    maximum_terms: int,
) -> tuple[int, RationalPolynomial]:
    """Validate the final PRS member and return its degree and leading unit."""

    from sympy import QQ, Poly

    variables = left.variables
    main_index = variables.index(main_variable)
    symbols = symbols_for_variables(variables)
    main_symbol = symbols[main_index]
    remaining_symbols = tuple(
        symbol for index, symbol in enumerate(symbols) if index != main_index
    )
    coefficient_field = QQ.frac_field(*remaining_symbols)

    def over_fraction_field(polynomial: RationalPolynomial) -> Any:
        return Poly(
            rational_polynomial_to_sympy(polynomial).as_expr(),
            main_symbol,
            domain=coefficient_field,
        )

    expected = over_fraction_field(left).gcd(over_fraction_field(right)).monic()
    actual = over_fraction_field(gcd_member).monic()
    if actual != expected:
        raise ValueError(
            "final subresultant member is not the fraction-field GCD associate"
        )

    return int(expected.degree()), polynomial_leading_coefficient_in_remaining_ring(
        gcd_member,
        main_variable,
        maximum_terms=maximum_terms,
    )


def polynomial_leading_coefficient_in_remaining_ring(
    polynomial: RationalPolynomial,
    main_variable: str,
    *,
    maximum_terms: int,
) -> RationalPolynomial:
    """Return the main-variable leading coefficient in ``QQ[rest]``."""

    from sympy import QQ, Poly

    variables = polynomial.variables
    main_index = variables.index(main_variable)
    remaining_variables = tuple(
        variable for index, variable in enumerate(variables) if index != main_index
    )
    remaining_symbols = symbols_for_variables(remaining_variables)
    univariate = _as_univariate_over_polynomial_ring(polynomial, main_variable)
    leading_coefficient = Poly(
        univariate.LC().as_expr(),
        *remaining_symbols,
        domain=QQ,
    )
    return rational_polynomial_from_sympy(
        leading_coefficient,
        remaining_variables,
        maximum_terms=maximum_terms,
    )


__all__ = [
    "MultivariatePrincipalSubresultantCoefficient",
    "MultivariateSubresultantMember",
    "MultivariateSubresultantSequenceRequest",
    "MultivariateSubresultantSequenceResult",
    "SubresultantSourceOrder",
    "fraction_field_gcd_relation",
    "polynomial_leading_coefficient_in_remaining_ring",
    "polynomial_resultant_in_remaining_ring",
    "polynomial_subresultant_sequence",
]
