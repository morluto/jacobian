"""Boundary cases for shared exact SymPy root-isolation primitives."""

from fractions import Fraction
from itertools import pairwise

import pytest
import sympy
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math._root_isolation import strict_root_count
from jacobian.math.number_theory.algebraic_numbers.real import RealAlgebraicValue
from jacobian.math.number_theory.algebraic_numbers.root_isolation import (
    compare_algebraic,
    isolate_real_roots,
)
from jacobian.math.number_theory.algebraic_numbers.root_isolation._models import (
    RootIsolationEntry,
    RootIsolationResult,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _polynomial() -> sympy.Poly:
    x = sympy.Symbol("x")
    return sympy.Poly(x * (x - 1) * (x - 2), x, domain=sympy.ZZ)


def test_strict_root_count_uses_singletons_for_rational_roots() -> None:
    polynomial = _polynomial()

    assert strict_root_count(polynomial, 1, 1) == 1
    assert (
        strict_root_count(polynomial, sympy.Rational(1, 2), sympy.Rational(1, 2)) == 0
    )


def test_strict_root_count_excludes_roots_at_open_interval_endpoints() -> None:
    polynomial = _polynomial()

    assert strict_root_count(polynomial, 0, 2) == 1
    assert strict_root_count(polynomial, -1, 1) == 1


def test_isolate_real_roots_accepts_canonical_rational_coefficients() -> None:
    coefficients = [
        CanonicalRational(num=1, den=1),
        CanonicalRational(num=0, den=1),
        CanonicalRational(num=-1, den=1),
    ]

    assert isolate_real_roots(coefficients) == [((-1, -1), 1), ((1, 1), 1)]


def _rationals(*values: int) -> list[CanonicalRational]:
    return [CanonicalRational(num=value, den=1) for value in values]


def _sympy_poly(coefficients: tuple[int, ...]) -> sympy.Poly:
    variable = sympy.Symbol("x")
    return sympy.Poly.from_list(list(coefficients), variable, domain=sympy.ZZ)


def _fraction(value: object) -> Fraction:
    rational = sympy.Rational(value)
    return Fraction(int(rational.p), int(rational.q))


def _algebraic(polynomial: tuple[int, ...], index: int) -> RealAlgebraicValue:
    return RealAlgebraicValue(polynomial=polynomial, real_root_index=index)


@pytest.mark.parametrize(
    ("coefficients", "expected_multiplicities"),
    [
        ((1, 0, -2), (1, 1)),
        ((10000, -300, 2), (1, 1)),
        ((10000, -100, 0), (1, 1)),
        ((1, -2, 1), (2,)),
        ((1, 0, 1), ()),
    ],
)
def test_isolated_intervals_each_hold_exactly_one_root_and_are_disjoint(
    coefficients: tuple[int, ...],
    expected_multiplicities: tuple[int, ...],
) -> None:
    intervals = isolate_real_roots(_rationals(*coefficients))

    assert tuple(multiplicity for _, multiplicity in intervals) == (
        expected_multiplicities
    )
    polynomial = _sympy_poly(coefficients)
    for (lower, upper), _multiplicity in intervals:
        assert strict_root_count(polynomial, lower, upper) == 1
    bounds = [
        (_fraction(lower), _fraction(upper), _fraction(lower) != _fraction(upper))
        for (lower, upper), _ in intervals
    ]
    for (_, upper, left_open), (next_lower, _, right_open) in pairwise(bounds):
        assert upper <= next_lower
        if upper == next_lower:
            assert left_open or right_open
    all_strict = all(
        upper < next_lower for (_, upper, _), (next_lower, _, _) in pairwise(bounds)
    )
    for position, ((lower, _upper), _multiplicity) in enumerate(intervals):
        below = int(polynomial.count_roots(-sympy.oo, lower))
        if polynomial.eval(lower) == 0:
            below -= 1
        assert below <= position
        if all_strict:
            assert below == position


def test_real_root_indexing_is_antisymmetric_and_transitive() -> None:
    negative_sqrt_two = _algebraic((1, 0, -2), 0)
    positive_sqrt_two = _algebraic((1, 0, -2), 1)
    cube_root_two = _algebraic((1, 0, 0, -2), 0)
    positive_sqrt_three = _algebraic((1, 0, -3), 1)

    assert compare_algebraic(negative_sqrt_two, positive_sqrt_two).order == "LT"
    assert compare_algebraic(positive_sqrt_two, negative_sqrt_two).order == "GT"
    assert compare_algebraic(positive_sqrt_two, positive_sqrt_two).order == "EQ"
    assert compare_algebraic(cube_root_two, positive_sqrt_two).order == "LT"
    assert compare_algebraic(positive_sqrt_two, positive_sqrt_three).order == "LT"
    assert compare_algebraic(cube_root_two, positive_sqrt_three).order == "LT"


def test_forged_overlapping_isolation_conclusion_is_rejected() -> None:
    source = RationalPolynomial(
        variables=("x",),
        polynomial=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=1, den=1), exponents=(2,)
                ),
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=-2, den=1), exponents=(0,)
                ),
            )
        ),
    )

    def _entry(lower: int, upper: int) -> RootIsolationEntry:
        return RootIsolationEntry(
            isolating_interval=(
                CanonicalRational(num=lower, den=1),
                CanonicalRational(num=upper, den=1),
            ),
            multiplicity=1,
            algebraic_value=RealAlgebraicValue.model_construct(
                polynomial=(1, 0, -2), real_root_index=0
            ),
        )

    with pytest.raises(ValidationError, match="disjoint"):
        RootIsolationResult(
            source_polynomial=source, roots=(_entry(0, 2), _entry(1, 3))
        )
