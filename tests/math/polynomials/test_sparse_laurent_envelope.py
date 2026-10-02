"""Sparse Laurent differentiation uses the canonical exponent envelope."""

import pytest
import sympy

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.polynomials._conversions import rational_function_from_sympy
from jacobian.math.polynomials.rational_functions.gradient import (
    RationalFunctionGradient,
    gradient,
)
from jacobian.math.polynomials.rational_functions.maps import jacobian_matrix
from jacobian.math.polynomials.rational_functions.values import RationalFunctionMap


@pytest.mark.parametrize("axes", [("x",), ("x", "y"), ("y", "x")])
@pytest.mark.parametrize("power", [64, 127])
def test_sparse_reciprocal_derivatives_fit_canonical_carrier(
    axes: tuple[str, ...], power: int
) -> None:
    x = sympy.Symbol("x")
    source = rational_function_from_sympy(x**-power, axes)
    expected = tuple(
        rational_function_from_sympy(
            -power * x ** (-power - 1) if axis == "x" else 0, axes
        )
        for axis in axes
    )
    result = gradient(source)
    restored = RationalFunctionGradient.model_validate_json(result.model_dump_json())
    assert restored.source == source
    assert restored.variables == axes
    assert restored.partial_derivatives == expected
    mapping = RationalFunctionMap(
        source_variables=axes, target_coordinates=("u",), components=(source,)
    )
    assert jacobian_matrix(mapping).entries == (expected,)


def test_sparse_numerator_survives_at_canonical_exponent_boundary() -> None:
    x, y = sympy.symbols("x y")
    source = rational_function_from_sympy(x**128 * y, ("x", "y", "unused"))
    expected = tuple(
        rational_function_from_sympy(value, source.variables)
        for value in (128 * x**127 * y, x**128, 0)
    )
    assert gradient(source).partial_derivatives == expected


def test_sparse_reciprocal_preserves_true_canonical_overflow() -> None:
    x = sympy.Symbol("x")
    source = rational_function_from_sympy(x**-128, ("x",))
    with pytest.raises(OperationResourceAdmissionError) as error:
        gradient(source)
    assert (
        error.value.errors()[0]["type"] == "rational_function.gradient.result_exponent"
    )
    mapping = RationalFunctionMap(
        source_variables=("x",), target_coordinates=("u",), components=(source,)
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        jacobian_matrix(mapping)
    assert (
        error.value.errors()[0]["type"]
        == "rational_function_map.jacobian.result_exponent"
    )


def test_sparse_eight_axis_reciprocal_boundary_avoids_expansion() -> None:
    axes = tuple(f"x{i}" for i in range(8))
    symbols = sympy.symbols("x0:8")
    expression = sympy.prod(variable**-127 for variable in symbols)
    source = rational_function_from_sympy(expression, axes)
    result = gradient(source)
    for axis, partial in enumerate(result.partial_derivatives):
        assert partial.numerator.terms[0].coefficient.num == -127
        assert partial.denominator.terms[0].exponents == tuple(
            128 if j == axis else 127 for j in range(8)
        )
        assert len(partial.numerator.terms) == len(partial.denominator.terms) == 1
        assert partial.variables == axes


def test_sparse_multi_term_laurent_support_stays_exact() -> None:
    x, y = sympy.symbols("x y")
    numerator = sympy.Rational(1, 2) * x**128 + sympy.Rational(1, 3) * x
    source = rational_function_from_sympy(numerator / y**127, ("x", "y"))
    expected = (
        rational_function_from_sympy(
            (64 * x**127 + sympy.Rational(1, 3)) / y**127, ("x", "y")
        ),
        rational_function_from_sympy(-127 * numerator / y**128, ("x", "y")),
    )
    assert gradient(source).partial_derivatives == expected
    assert all(len(partial.numerator.terms) == 2 for partial in expected)
