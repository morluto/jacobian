"""Property tests for exact real-algebraic order.

Order is a signed relation, so the defining evidence is the order axioms
(reflexivity, antisymmetry, transitivity) together with an independent
high-precision numeric sign of the difference.  A flipped comparison cannot
satisfy both.
"""

from __future__ import annotations

import sympy
from hypothesis import given, settings
from hypothesis import strategies as st

from jacobian.math.number_theory.algebraic_numbers.real import RealAlgebraicValue
from jacobian.math.number_theory.algebraic_numbers.root_isolation import (
    compare_algebraic,
)

_VARIABLE = sympy.Symbol("x")
_CASE = st.tuples(
    st.lists(st.integers(min_value=-5, max_value=5), min_size=1, max_size=3),
    st.integers(min_value=0, max_value=7),
)
_ORDERS = {"LT": -1, "EQ": 0, "GT": 1}


def _value_and_root(case: tuple[list[int], int]):
    coefficients = (1, *case[0])
    polynomial = sympy.Poly.from_list(list(coefficients), _VARIABLE, domain=sympy.ZZ)
    if polynomial.is_irreducible is not True:
        return None
    roots = polynomial.real_roots()
    if not roots:
        return None
    index = case[1] % len(roots)
    value = RealAlgebraicValue._from_admitted_polynomial(
        polynomial=coefficients,
        real_root_index=index,
    )
    return value, sympy.N(roots[index], 60)


def _opposite(order: str) -> str:
    return {"LT": "GT", "EQ": "EQ", "GT": "LT"}[order]


@settings(max_examples=40, deadline=None)
@given(first=_CASE, second=_CASE, third=_CASE)
def test_real_algebraic_order_laws_and_numeric_oracle(
    first: tuple[list[int], int],
    second: tuple[list[int], int],
    third: tuple[list[int], int],
) -> None:
    left = _value_and_root(first)
    middle = _value_and_root(second)
    right = _value_and_root(third)
    if left is None or middle is None or right is None:
        return

    left_value, left_root = left
    middle_value, middle_root = middle
    right_value = right[0]

    assert compare_algebraic(left_value, left_value).order == "EQ"

    forward = compare_algebraic(left_value, middle_value).order
    assert compare_algebraic(middle_value, left_value).order == _opposite(forward)

    difference = left_root - middle_root
    threshold = sympy.Float("1e-40")
    if difference > threshold:
        assert forward == "GT"
    elif difference < -threshold:
        assert forward == "LT"

    forward_sign = _ORDERS[forward]
    second_sign = _ORDERS[compare_algebraic(middle_value, right_value).order]
    total_sign = _ORDERS[compare_algebraic(left_value, right_value).order]
    if forward_sign <= 0 and second_sign <= 0:
        assert total_sign <= 0
    if forward_sign >= 0 and second_sign >= 0:
        assert total_sign >= 0
