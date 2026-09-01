"""Conservative exact-CAD work bounds for plane component profiles."""

from __future__ import annotations

from math import factorial, lcm

from jacobian.math.polynomials.real_algebra._plane_component_models import (
    MAX_PLANE_COMPONENT_POINT_COEFFICIENT_DIGITS,
    MAX_PLANE_COMPONENT_POINT_DEGREE,
)
from jacobian.math.polynomials.values import RationalPolynomial

MAX_PLANE_COMPONENT_PROJECTION_DEGREE_SUM = 2_048
MAX_PLANE_COMPONENT_PREDICTED_CELLS = 250_000
# Pairing two degree-sixteen, 512-digit coordinate markers is the largest
# declared refinement resultant: each half contributes degree * height and the
# Sylvester determinant contributes at most (2 * degree)! terms.
MAX_PLANE_COMPONENT_PROJECTED_COEFFICIENT_DIGITS = (
    2 * MAX_PLANE_COMPONENT_POINT_DEGREE * MAX_PLANE_COMPONENT_POINT_COEFFICIENT_DIGITS
    + len(str(factorial(2 * MAX_PLANE_COMPONENT_POINT_DEGREE)))
)


def _cleared_coefficient_digits(polynomial: RationalPolynomial) -> int:
    denominators = tuple(
        term.coefficient.as_fraction().denominator
        for term in polynomial.polynomial.terms
    )
    common_denominator = lcm(*denominators) if denominators else 1
    return max(
        (
            len(
                str(
                    abs(
                        term.coefficient.as_fraction().numerator
                        * (
                            common_denominator
                            // term.coefficient.as_fraction().denominator
                        )
                    )
                )
            )
            for term in polynomial.polynomial.terms
        ),
        default=1,
    )


def plane_projection_bound(
    polynomials: tuple[RationalPolynomial, ...],
) -> tuple[int, int]:
    """Bound modified-McCallum projection degree and lifted cell count."""

    degrees = tuple(
        (
            max(
                (term.exponents[0] for term in polynomial.polynomial.terms),
                default=0,
            ),
            max(
                (term.exponents[1] for term in polynomial.polynomial.terms),
                default=0,
            ),
        )
        for polynomial in polynomials
    )
    projected_degree_sum = 0
    for degree_x, degree_y in degrees:
        projected_degree_sum += (degree_y + 1) * degree_x
        projected_degree_sum += max(0, 2 * degree_y - 2) * degree_x
    for left_index, (left_x, left_y) in enumerate(degrees):
        for right_x, right_y in degrees[left_index + 1 :]:
            projected_degree_sum += left_y * right_x + right_y * left_x
    x_cells = 2 * projected_degree_sum + 1
    y_cells = 2 * sum(degree_y for _degree_x, degree_y in degrees) + 1
    return projected_degree_sum, x_cells * y_cells


def _projection_coefficient_bound(
    polynomials: tuple[RationalPolynomial, ...],
    *,
    swap_axes: bool,
) -> int:
    """Bound one ordered projection family's coefficient height.

    Modified McCallum projection uses coefficients, discriminants, and pairwise
    resultants. These determinant bounds count every Sylvester permutation and
    every possible convolution of coefficient polynomials in ``x``. They also
    cover fraction-free determinant minors used to form those polynomials.
    """

    data = tuple(
        (
            max(
                (
                    term.exponents[1 if swap_axes else 0]
                    for term in polynomial.polynomial.terms
                ),
                default=0,
            ),
            max(
                (
                    term.exponents[0 if swap_axes else 1]
                    for term in polynomial.polynomial.terms
                ),
                default=0,
            ),
            _cleared_coefficient_digits(polynomial),
        )
        for polynomial in polynomials
    )
    bound = max((height for _degree_x, _degree_y, height in data), default=1)
    for degree_x, degree_y, height in data:
        if degree_y < 2:
            continue
        determinant_order = 2 * degree_y - 1
        derivative_height = height + len(str(degree_y))
        term_digits = (degree_y - 1) * height + degree_y * derivative_height
        multiplicity = factorial(determinant_order) * (degree_x + 1) ** (
            determinant_order
        )
        bound = max(bound, term_digits + len(str(multiplicity)))
    for left_index, (left_x, left_y, left_height) in enumerate(data):
        for right_x, right_y, right_height in data[left_index + 1 :]:
            if left_y == 0 or right_y == 0:
                continue
            determinant_order = left_y + right_y
            term_digits = right_y * left_height + left_y * right_height
            multiplicity = (
                factorial(determinant_order)
                * (left_x + 1) ** right_y
                * (right_x + 1) ** left_y
            )
            bound = max(bound, term_digits + len(str(multiplicity)))
    return bound


def plane_projection_coefficient_bound(
    polynomials: tuple[RationalPolynomial, ...],
) -> int:
    """Bound decimal height after eliminating the second coordinate."""

    return _projection_coefficient_bound(polynomials, swap_axes=False)


__all__ = [
    "MAX_PLANE_COMPONENT_PREDICTED_CELLS",
    "MAX_PLANE_COMPONENT_PROJECTED_COEFFICIENT_DIGITS",
    "MAX_PLANE_COMPONENT_PROJECTION_DEGREE_SUM",
    "plane_projection_bound",
    "plane_projection_coefficient_bound",
]
