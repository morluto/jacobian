from __future__ import annotations

import json
from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials._conversions import rational_function_from_sympy
from jacobian.math.polynomials.local_series import (
    rational_function_at_infinity,
    rational_function_at_point,
)
from jacobian.math.polynomials.local_series._tools import TOOLS
from jacobian.math.polynomials.local_series.arithmetic import add


def test_simple_pole_and_repeated_pole_have_exact_laurent_prefixes() -> None:
    import sympy as sp

    x = sp.symbols("x")
    simple = rational_function_from_sympy(1 / (x - 1), ("x",))
    repeated = rational_function_from_sympy((x + 1) / (x - 1) ** 2, ("x",))

    simple_result = rational_function_at_point(
        simple, CanonicalRational(num=1, den=1), 3
    )
    assert (simple_result.series.valuation_lower, simple_result.series.precision) == (
        -1,
        3,
    )
    assert tuple(c.as_fraction() for c in simple_result.series.coefficients) == (
        1,
        0,
        0,
        0,
    )
    assert (simple_result.numerator_order, simple_result.denominator_order) == (0, 1)
    assert simple_result.valuation == -1
    assert simple_result.pole_order == 1
    assert simple_result.normalized_unit_quotient is not None
    assert simple_result.product_residual_precision == 4

    repeated_result = rational_function_at_point(
        repeated, CanonicalRational(num=1, den=1), 2
    )
    assert (
        repeated_result.series.valuation_lower,
        repeated_result.series.precision,
    ) == (-2, 2)
    assert tuple(c.as_fraction() for c in repeated_result.series.coefficients) == (
        2,
        1,
        0,
        0,
    )


def test_regular_expansion_at_nonzero_center_and_cutoff_past_valuation() -> None:
    import sympy as sp

    x = sp.symbols("x")
    function = rational_function_from_sympy((x**2 + 1) / (x + 1), ("x",))
    result = rational_function_at_point(function, CanonicalRational(num=1, den=1), 4)

    # (2 + 2t + t^2)/(2 + t) = 1 + t/2 + t^2/4 - t^3/8 + O(t^4).
    assert (result.series.valuation_lower, result.series.precision) == (0, 4)
    assert tuple(c.as_fraction() for c in result.series.coefficients) == (
        Fraction(1),
        Fraction(1, 2),
        Fraction(1, 4),
        Fraction(-1, 8),
    )

    high_order_zero = rational_function_from_sympy((x - 2) ** 4 / (x + 1), ("x",))
    zero_prefix = rational_function_at_point(
        high_order_zero, CanonicalRational(num=2, den=1), 3
    )
    assert (zero_prefix.series.valuation_lower, zero_prefix.series.precision) == (0, 3)
    assert tuple(c.as_fraction() for c in zero_prefix.series.coefficients) == (0, 0, 0)
    assert zero_prefix.valuation == 4
    assert zero_prefix.zero_order == 4


def test_zero_function_canceled_orders_and_rational_center() -> None:
    import sympy as sp

    x = sp.symbols("x")
    zero = rational_function_from_sympy(sp.Integer(0), ("x",))
    canceled = rational_function_from_sympy((x - 1) / (x - 1), ("x",))
    regular = rational_function_from_sympy(1 / (x + 1), ("x",))

    zero_result = rational_function_at_point(zero, CanonicalRational(num=0, den=1), 3)
    assert tuple(value.as_fraction() for value in zero_result.series.coefficients) == (
        0,
        0,
        0,
    )
    assert zero_result.valuation is None
    assert (
        rational_function_at_point(canceled, CanonicalRational(num=1, den=1), 3)
        .series.coefficients[0]
        .as_fraction()
        == 1
    )

    center = CanonicalRational(num=1, den=2)
    result = rational_function_at_point(regular, center, 3)
    assert result.series.center == center
    assert tuple(value.as_fraction() for value in result.series.coefficients) == (
        Fraction(2, 3),
        Fraction(-4, 9),
        Fraction(8, 27),
    )


def test_tool_is_published_and_coefficient_work_is_bounded() -> None:
    tool = next(
        item
        for item in TOOLS
        if item.operation_id == "local_series.from_rational_function_at_point.compute"
    )
    request = tool.request_type.model_validate_json(json.dumps(tool.examples[0].input))
    result = tool.run(request)
    assert result.series.valuation_lower == -1
    assert result.series.coefficients[0].as_fraction() == 1

    import sympy as sp

    x = sp.symbols("x")
    function = rational_function_from_sympy(1 / x, ("x",))
    with pytest.raises(OperationResourceAdmissionError):
        rational_function_at_point(function, CanonicalRational(num=0, den=1), 4097)


def test_infinity_expansion_records_reciprocal_parent_and_exact_growth_order() -> None:
    import sympy as sp

    x = sp.symbols("x")
    polynomial = rational_function_from_sympy(x**2 + 1, ("x",))
    proper = rational_function_from_sympy((x + 1) / (x**2 + 1), ("x",))

    result = rational_function_at_infinity(polynomial, 4)
    assert result.series.place == "INFINITY"
    assert result.series.center == CanonicalRational(num=0, den=1)
    assert (result.series.valuation_lower, result.series.precision) == (-2, 4)
    assert tuple(c.as_fraction() for c in result.series.coefficients) == (
        1,
        0,
        1,
        0,
        0,
        0,
    )
    assert (result.numerator_order, result.denominator_order) == (-2, 0)
    assert result.valuation == -2
    assert result.pole_order == 2
    assert result.product_residual_precision == 4
    assert result.model_validate_json(result.model_dump_json()) == result
    assert add(result.series, result.series).place == "INFINITY"

    proper_result = rational_function_at_infinity(proper, 3)
    assert (proper_result.series.valuation_lower, proper_result.series.precision) == (
        1,
        3,
    )
    assert tuple(c.as_fraction() for c in proper_result.series.coefficients) == (1, 1)

    tool = next(
        item
        for item in TOOLS
        if item.operation_id == "local_series.at_infinity.compute"
    )
    request = tool.request_type.model_validate_json(json.dumps(tool.examples[0].input))
    assert tool.run(request).series.place == "INFINITY"

    finite = rational_function_at_point(
        rational_function_from_sympy(1 / (x - 1), ("x",)),
        CanonicalRational(num=0, den=1),
        2,
    )
    with pytest.raises(OperationDomainValidationError):
        add(finite.series, result.series)
