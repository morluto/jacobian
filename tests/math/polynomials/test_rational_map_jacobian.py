"""Exact map differentiation, retained domains and whole-matrix admission."""

from time import monotonic

import pytest
from pydantic import ValidationError
from pydantic_core import PydanticCustomError
from sympy import QQ, Poly, symbols

from jacobian._execution import OperationExecutionTimeoutError, request_execution
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.polynomials._conversions import rational_function_from_sympy
from jacobian.math.polynomials.rational_functions import RationalFunctionMap
from jacobian.math.polynomials.rational_functions.maps import (
    RationalFunctionMapJacobian,
    jacobian_matrix,
)
from jacobian.math.polynomials.values import RationalFunction


def test_issue_example_and_unchanged_serialized_components() -> None:
    x, y = symbols("x y")
    expressions = ((x * x + y) / (x - y), (x + y) / (x * x + 1))
    source = RationalFunctionMap(
        source_variables=("x", "y"),
        target_coordinates=("u", "v"),
        components=tuple(
            rational_function_from_sympy(e, ("x", "y")) for e in expressions
        ),
    )
    result = jacobian_matrix(source)
    assert result.source == source
    assert result.row_axis == ("u", "v")
    assert result.column_axis == ("x", "y")
    for row, expression in zip(result.entries, expressions, strict=True):
        for value, axis in zip(row, (x, y), strict=True):
            assert value == rational_function_from_sympy(
                expression.diff(axis), ("x", "y")
            )
    assert (
        RationalFunctionMapJacobian.model_validate_json(result.model_dump_json())
        == result
    )
    consumer = RationalFunctionMap(
        source_variables=result.column_axis,
        target_coordinates=("a", "b"),
        components=tuple(
            RationalFunction.model_validate_json(v.model_dump_json())
            for v in result.entries[0]
        ),
    )
    assert (
        jacobian_matrix(consumer).entries[0][1]
        == jacobian_matrix(consumer).entries[1][0]
    )


@pytest.mark.parametrize("axes", [(), ("x", "y")])
@pytest.mark.parametrize("rows", [0, 3])
def test_empty_dimensions_and_constants(axes: tuple[str, ...], rows: int) -> None:
    source = RationalFunctionMap(
        source_variables=axes,
        target_coordinates=tuple(f"u{i}" for i in range(rows)),
        components=tuple(rational_function_from_sympy(i, axes) for i in range(rows)),
    )
    result = jacobian_matrix(source)
    assert len(result.entries) == rows
    assert all(len(row) == len(axes) for row in result.entries)
    assert all(not value.numerator.terms for row in result.entries for value in row)


def test_permuted_axes_and_original_common_locus() -> None:
    x, y = symbols("x y")
    axes = ("y", "x")
    source = RationalFunctionMap(
        source_variables=axes,
        target_coordinates=("v", "u"),
        components=tuple(rational_function_from_sympy(e, axes) for e in (1 / x, 1 / y)),
    )
    result = jacobian_matrix(source)
    assert not result.entries[0][0].numerator.terms
    assert not result.entries[1][1].numerator.terms
    assert result.source.components == source.components
    assert result.entries[0][1] == rational_function_from_sympy(-1 / x**2, axes)
    assert result.entries[1][0] == rational_function_from_sympy(-1 / y**2, axes)


def test_many_sparse_rows_and_eight_axis_support() -> None:
    axes = tuple(f"x{i}" for i in range(8))
    variables = symbols(" ".join(axes))
    expression = 1
    for variable in variables:
        expression *= variable**64
    component = rational_function_from_sympy(expression, axes)
    source = RationalFunctionMap(
        source_variables=axes,
        target_coordinates=tuple(f"u{i}" for i in range(1024)),
        components=(component,) * 1024,
    )
    result = jacobian_matrix(source)
    assert len(result.entries) == 1024
    assert result.entries[0] == result.entries[-1]
    assert result.entries[0][0].numerator.terms[0].coefficient.num == 64


def test_aggregate_output_allocation_rejects_whole_map() -> None:
    x, y = symbols("x y")
    # Every scalar row is cheap and within the canonical carrier. Their
    # combined derivative support exceeds the whole-map allocation.
    component = rational_function_from_sympy(
        sum(x**i * y ** (63 - i) for i in range(64)), ("x", "y")
    )
    source = RationalFunctionMap(
        source_variables=("x", "y"),
        target_coordinates=tuple(f"u{i}" for i in range(600)),
        components=(component,) * 600,
    )
    with pytest.raises(OperationResourceAdmissionError, match="65,536"):
        jacobian_matrix(source)


def test_authored_noncanonical_component_and_malformed_axes() -> None:
    x = symbols("x")
    p = rational_function_from_sympy(x, ("x",)).numerator
    invalid = RationalFunction(variables=("x",), numerator=p, denominator=p)
    source = RationalFunctionMap(
        source_variables=("x",), target_coordinates=("u",), components=(invalid,)
    )
    with pytest.raises(PydanticCustomError, match="coprime"):
        jacobian_matrix(source)
    with pytest.raises(ValidationError, match="ordered source axis"):
        RationalFunctionMap(
            source_variables=("y",), target_coordinates=("u",), components=(invalid,)
        )
    with pytest.raises(ValidationError, match="distinct"):
        RationalFunctionMap(
            source_variables=("x",),
            target_coordinates=("u", "u"),
            components=(invalid, invalid),
        )


def test_caller_deadline_and_output_height_recovery() -> None:
    x = symbols("x")
    source = RationalFunctionMap(
        source_variables=("x",),
        target_coordinates=("u",),
        components=(rational_function_from_sympy(10**127 * x**64, ("x",)),),
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        jacobian_matrix(source)
    assert (
        error.value.errors()[0]["type"]
        == "rational_function_map.jacobian.result_height"
    )
    with (
        request_execution(monotonic() - 100),
        pytest.raises(OperationExecutionTimeoutError),
    ):
        jacobian_matrix(source)


def test_conic_parametrization_components_compose_unchanged() -> None:
    from jacobian._exact import CanonicalRational
    from jacobian.math.geometry.algebraic_curves.operations import (
        rational_conic_parametrization,
    )
    from jacobian.math.polynomials._conversions import rational_polynomial_from_sympy
    from jacobian.math.polynomials.maps._models import VariablePoint

    x, y = symbols("x y")
    conic = rational_conic_parametrization(
        rational_polynomial_from_sympy(
            Poly(x * x + y * y - 1, x, y, domain=QQ), ("x", "y")
        ),
        VariablePoint(
            variables=("x", "y"),
            values=(CanonicalRational(num=1, den=1), CanonicalRational(num=0, den=1)),
        ),
        "t",
    )
    source = RationalFunctionMap(
        source_variables=("t",),
        target_coordinates=("x", "y"),
        components=conic.coordinates,
    )
    result = jacobian_matrix(
        RationalFunctionMap.model_validate_json(source.model_dump_json())
    )
    assert result.source.components == conic.coordinates
    assert len(result.entries) == 2
    assert all(len(row) == 1 for row in result.entries)


def test_quotient_rule_by_independent_exact_coefficient_convolution() -> None:
    from fractions import Fraction
    from random import Random

    from jacobian.math.polynomials.values import SparseRationalPolynomial

    type Coefficients = dict[tuple[int, ...], Fraction]

    def coefficients(p: SparseRationalPolynomial) -> Coefficients:
        return {t.exponents: t.coefficient.as_fraction() for t in p.terms}

    def product(a: Coefficients, b: Coefficients) -> Coefficients:
        out: Coefficients = {}
        for e, c in a.items():
            for f, d in b.items():
                key = tuple(i + j for i, j in zip(e, f, strict=True))
                out[key] = out.get(key, Fraction(0)) + c * d
        return {e: c for e, c in out.items() if c}

    def derivative(p: Coefficients, axis: int) -> Coefficients:
        return {
            tuple(n - int(i == axis) for i, n in enumerate(e)): c * e[axis]
            for e, c in p.items()
            if e[axis]
        }

    rng = Random(2879)
    x, y = symbols("x y")
    components = tuple(
        rational_function_from_sympy(
            (rng.randrange(1, 5) * x * x + y + i) / (x + y + i + 1), ("x", "y")
        )
        for i in range(12)
    )
    source = RationalFunctionMap(
        source_variables=("x", "y"),
        target_coordinates=tuple(f"u{i}" for i in range(12)),
        components=components,
    )
    result = jacobian_matrix(source)
    for component, row in zip(components, result.entries, strict=True):
        p, q = coefficients(component.numerator), coefficients(component.denominator)
        for axis, value in enumerate(row):
            left = product(derivative(p, axis), q)
            right = product(p, derivative(q, axis))
            quotient = {
                e: c
                for e in left.keys() | right.keys()
                if (c := left.get(e, Fraction(0)) - right.get(e, Fraction(0)))
            }
            assert product(coefficients(value.numerator), product(q, q)) == product(
                quotient, coefficients(value.denominator)
            )
