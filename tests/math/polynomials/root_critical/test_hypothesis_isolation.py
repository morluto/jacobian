"""Property-based isolation checks for root-critical rectangle construction.

The strategies deliberately *construct* degenerate structure (near-equal
quadratic surds and dense low-degree supports) instead of sampling uniformly,
because a uniformly random draw will not hit the close-root gaps that expose
partial-obstacle-set bugs.
"""

from __future__ import annotations

import sympy
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.math.polynomials.root_critical._isolation_invariants import (
    require_axis_rectangles_are_isolating,
)

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.polynomials.root_critical.operations import _family

z = sympy.Symbol("z")

_NEAR_COLLISION_SETTINGS = settings(max_examples=25, deadline=None)


@_NEAR_COLLISION_SETTINGS
@given(
    base=st.integers(min_value=1, max_value=4),
    exponent=st.integers(min_value=5, max_value=80),
)
def test_close_quadratic_surds_are_isolating(base: int, exponent: int) -> None:
    """``sqrt(base)`` and ``sqrt(base + 1/q)`` must never share a rectangle."""
    denominator = 10**exponent
    sibling = sympy.Rational(base * denominator + 1, denominator)
    polynomial = sympy.Poly((z**2 - base) * (z**2 - sibling), z, domain="QQ")
    records, _values = _family(polynomial)
    require_axis_rectangles_are_isolating(
        polynomial.sqf_part(), tuple(record.rectangle for record in records)
    )


@settings(max_examples=25, deadline=None)
@given(
    coefficients=st.lists(
        st.integers(min_value=-6, max_value=6),
        min_size=2,
        max_size=4,
    ).filter(any)
)
def test_dense_low_degree_supports_are_isolating(coefficients: list[int]) -> None:
    """Dense cubic and quadratic supports keep every rectangle isolating."""
    polynomial = sympy.Poly.from_list(coefficients, z, domain=sympy.ZZ)
    if polynomial.degree() <= 0:
        return
    try:
        records, _values = _family(polynomial)
    except OperationResourceAdmissionError as error:
        # Only the documented unsupported backend root form is outside this
        # property. Other admission failures must remain visible.
        if (
            error.errors()[0]["type"]
            != "polynomial.root_critical.root_carrier_backend_form"
        ):
            raise
        return
    require_axis_rectangles_are_isolating(
        polynomial.sqf_part(), tuple(record.rectangle for record in records)
    )
