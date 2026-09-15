"""Property tests for the exact multivariate resultant orientation.

The resultant is a signed invariant, so the defining evidence is relational:
the argument-swap law ``Res(f, g) = (-1)**(deg f * deg g) Res(g, f)`` and an
independent Sylvester-determinant oracle.  This mirrors the failure mode of
upstream sympy/sympy#10666, where a degree-canonicalizing PRS dropped the swap
sign only for odd-degree pairs.
"""

from __future__ import annotations

from fractions import Fraction

import sympy
from hypothesis import example, given, settings
from hypothesis import strategies as st

from jacobian.math.polynomials.multivariate._resultant import (
    MultivariateResultantRequest,
)
from jacobian.math.polynomials.multivariate._tools import _compute_resultant
from jacobian.math.polynomials.values import RationalPolynomial

_COEFFICIENTS = st.integers(min_value=-6, max_value=6)


def _univariate(coefficients: list[int]) -> RationalPolynomial:
    """Build a ``QQ[x, y]`` polynomial of ``y``-degree zero.

    The multivariate contract requires at least two variables; keeping the
    remaining variable at degree zero makes the resultant scalar-valued while
    exercising the exact Sylvester orientation.
    """

    degree = len(coefficients) - 1
    terms = [
        {
            "coefficient": {"num": int(coefficient), "den": 1},
            "exponents": [degree - index, 0],
        }
        for index, coefficient in enumerate(coefficients)
        if coefficient
    ]
    return RationalPolynomial.model_validate(
        {
            "domain": "QQ",
            "variables": ["x", "y"],
            "polynomial": {"terms": terms},
        }
    )


def _descending_coefficients(polynomial: RationalPolynomial) -> list[Fraction] | None:
    terms = polynomial.polynomial.terms
    if not terms:
        return None
    degree = max(term.exponents[0] for term in terms)
    coefficients: list[Fraction] = [Fraction(0)] * (degree + 1)
    for term in terms:
        coefficients[degree - term.exponents[0]] = term.coefficient.as_fraction()
    return coefficients


def _sylvester_determinant(
    left: list[Fraction] | None, right: list[Fraction] | None
) -> sympy.Expr:
    """The exact Sylvester determinant, independent of the PRS backend."""

    if left is None or right is None:
        return sympy.Integer(0)
    m = len(left) - 1
    n = len(right) - 1
    if m == 0 and n == 0:
        return sympy.Integer(1)
    size = m + n
    rows: list[list[sympy.Expr]] = []
    for index in range(n):
        rows.append([0] * index + list(left) + [0] * (n - 1 - index))
    for index in range(m):
        rows.append([0] * index + list(right) + [0] * (m - 1 - index))
    assert all(len(row) == size for row in rows)
    return sympy.Matrix(rows).det()


def _resultant_value(left: RationalPolynomial, right: RationalPolynomial) -> Fraction:
    result = _compute_resultant(
        MultivariateResultantRequest(left=left, right=right, elimination_variable="x")
    )
    if result.resultant.kind == "SCALAR":
        return result.resultant.value.as_fraction()
    # The remaining variable is retained at degree zero, so every term must be
    # constant in it and the scalar is their sum.
    return sum(
        (
            term.coefficient.as_fraction()
            for term in result.resultant.value.polynomial.terms
        ),
        Fraction(0),
    )


@settings(max_examples=60, deadline=None)
@given(
    left_coefficients=st.lists(_COEFFICIENTS, min_size=1, max_size=5),
    right_coefficients=st.lists(_COEFFICIENTS, min_size=1, max_size=5),
)
@example(left_coefficients=[2, 1], right_coefficients=[1, 0, 0, 0])
@example(left_coefficients=[1, 0, 0, 0], right_coefficients=[2, 1])
@example(left_coefficients=[0], right_coefficients=[1, 0, 0, 0])
def test_resultant_swap_law_and_sylvester_oracle(
    left_coefficients: list[int], right_coefficients: list[int]
) -> None:
    left = _univariate(left_coefficients)
    right = _univariate(right_coefficients)

    forward = _resultant_value(left, right)
    reverse = _resultant_value(right, left)

    left_degree = max((term.exponents[0] for term in left.polynomial.terms), default=0)
    right_degree = max(
        (term.exponents[0] for term in right.polynomial.terms), default=0
    )
    if left.polynomial.terms and right.polynomial.terms:
        # Two nonzero inputs: the sign law is exact.
        sign = -1 if (left_degree * right_degree) % 2 else 1
        assert forward == sign * reverse
    else:
        assert forward == 0
        assert reverse == 0

    expected = _sylvester_determinant(
        _descending_coefficients(left), _descending_coefficients(right)
    )
    assert forward == Fraction(expected)
