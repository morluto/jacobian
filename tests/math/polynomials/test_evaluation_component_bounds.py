"""Source-derived evaluation bounds dominate independent rational arithmetic."""

from fractions import Fraction
from itertools import product

import pytest

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.math.polynomials._elementary_kernel import rational_polynomial_evaluate
from jacobian.math.polynomials.maps._models import VariablePoint
from jacobian.math.polynomials.maps.operations import evaluate_polynomial
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
    rational_evaluation_component_digit_bounds,
)


def _polynomial(
    variables: tuple[str, ...], entries: tuple[tuple[tuple[int, ...], Fraction], ...]
) -> RationalPolynomial:
    return RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    exponents=exponents,
                    coefficient=CanonicalRational.from_fraction(coefficient),
                )
                for exponents, coefficient in entries
                if coefficient
            )
        ),
    )


def test_bound_dominates_multiaxis_fraction_evaluation() -> None:
    support = ((3, 1), (1, 2), (0, 0))
    scalars = (Fraction(0), Fraction(1), Fraction(-1), Fraction(2, 3), Fraction(-3, 5))
    for coefficients in product(
        (Fraction(1), Fraction(1, 2), Fraction(-3, 5)), repeat=3
    ):
        polynomial = _polynomial(
            ("x", "y"), tuple(zip(support, coefficients, strict=True))
        )
        for coordinates in product(scalars, repeat=2):
            point = tuple(
                CanonicalRational.from_fraction(value) for value in coordinates
            )
            bounds = rational_evaluation_component_digit_bounds(polynomial, point)
            exact = sum(
                (
                    coefficient
                    * coordinates[0] ** exponents[0]
                    * coordinates[1] ** exponents[1]
                    for exponents, coefficient in zip(
                        support, coefficients, strict=True
                    )
                ),
                Fraction(0),
            )
            assert len(str(abs(exact.numerator))) <= bounds[0]
            assert len(str(exact.denominator)) <= bounds[1]


def test_zero_terms_and_empty_axis_constants_keep_small_bounds() -> None:
    zero = _polynomial(("x",), ())
    assert rational_evaluation_component_digit_bounds(
        zero, (CanonicalRational(num=1, den=97),)
    ) == (1, 1)
    annihilated = _polynomial(("x",), (((3,), Fraction(1, 97)),))
    assert rational_evaluation_component_digit_bounds(
        annihilated, (CanonicalRational(num=0, den=1),)
    ) == (1, 1)
    constant = _polynomial((), (((), Fraction(-3, 97)),))
    assert rational_evaluation_component_digit_bounds(constant, ()) == (1, 2)


@pytest.mark.parametrize("balanced", (False, True))
def test_native_identity_evaluators_accept_maximum_width_points(balanced: bool) -> None:
    magnitude = 10 ** (MAX_CANONICAL_RATIONAL_DIGITS - 1)
    point = CanonicalRational(num=magnitude if balanced else 1, den=magnitude + 1)
    polynomial = _polynomial(("x",), (((1,), Fraction(1)),))
    assert rational_polynomial_evaluate(polynomial, point).value == point
    assert (
        evaluate_polynomial(
            polynomial, VariablePoint(variables=("x",), values=(point,))
        ).value
        == point
    )
