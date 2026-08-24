"""Private SymPy adapter for exact polynomial subresultant sequences."""

from __future__ import annotations

from typing import Any, Literal

from jacobian.math.polynomials._conversions import (
    rational_polynomial_from_sympy,
    rational_polynomial_to_sympy,
    symbols_for_variables,
)
from jacobian.math.polynomials.values import RationalPolynomial

SubresultantSourceOrder = Literal["LEFT_RIGHT", "RIGHT_LEFT"]


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
    "SubresultantSourceOrder",
    "fraction_field_gcd_relation",
    "polynomial_leading_coefficient_in_remaining_ring",
    "polynomial_resultant_in_remaining_ring",
    "polynomial_subresultant_sequence",
]
